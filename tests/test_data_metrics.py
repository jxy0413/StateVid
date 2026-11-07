"""Temporary synthetic fixtures verify label semantics and objective reductions."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from statevid.data import (ChangeDataset,build_wordmap,collate_batch,
                                      decode_binary_mask,decode_caption,encode_caption)
from statevid.losses import task_loss
from statevid.metrics import (BinaryMetrics,SemanticMetrics,DamageMetrics,
                                         caption_scores,create_metric,update_metrics)


def fixture_manifest(tmp_path,task="bcd",shapes=((9,11),(12,8))):
    records = []
    for i,(h,w) in enumerate(shapes):
        folder = tmp_path/str(i)
        folder.mkdir()
        rgb = np.full((h,w,3),127,np.uint8)
        label = np.zeros((h,w),np.uint8);label[:h//2,:w//2] = 1
        row = {"id":str(i)}
        values = {"pre":rgb,"post":rgb.copy()}
        if task == "bcd":values["label"] = label*255
        elif task == "scd":values.update(label_pre=label,label_post=label*2)
        elif task == "bda":values.update(localization=label,damage=label)
        else:row["captions"] = [["a","new","house","appears","here"],["a","building","appears"]]
        for key,value in values.items():
            path = folder/(key+".png");Image.fromarray(value).save(path)
            row[key] = path.relative_to(tmp_path).as_posix()
        records.append(row)
    data = {"task":task,"dataset":"Fixture","splits":{"train":records,"val":[],"test":[]}}
    if task == "scd":data["num_classes"] = 3
    path = tmp_path/"manifest.json";path.write_text(json.dumps(data))
    return path,data


def test_binary_encoding_and_ignore_are_explicit():
    array = np.array([[0,255],[255,0]],np.uint8)
    target,valid = decode_binary_mask(array)
    assert target.tolist() == [[0,1],[1,0]] and valid.all()
    target,valid = decode_binary_mask(np.array([[0,1,255]],np.uint8),ignore_index=255)
    assert target.tolist() == [[0,1,0]] and valid.tolist() == [[True,True,False]]
    with pytest.raises(ValueError):decode_binary_mask(np.array([[0,1,255]],np.uint8))


def test_native_padding_excludes_targets(tmp_path):
    path,_ = fixture_manifest(tmp_path)
    ds = ChangeDataset(path,"train",augment=False)
    sample = ds[0]
    assert sample["pre"].shape == (3,9,11)
    batch = collate_batch([sample,ds[1]])
    assert batch["pre"].shape == (2,3,16,16)
    assert batch["valid_mask"].sum().item() == 9*11+12*8
    assert batch["label"].sum().item() == 4*5+6*4


def test_geometry_is_synchronized(tmp_path):
    path,_ = fixture_manifest(tmp_path,shapes=((12,12),))
    ds = ChangeDataset(path,"train",crop_size=8)
    for _ in range(5):
        item = ds[0]
        assert torch.equal(item["pre"],item["post"])
        assert item["label"].shape == (1,8,8)
        assert item["valid_mask"].all()


def test_scd_valid_mask_and_cached_pseudo_labels(tmp_path):
    path,data = fixture_manifest(tmp_path,"scd",shapes=((9,11),))
    row = data["splits"]["train"][0]
    value = np.full((9,11),255,np.uint8);value[6:,6:] = 2
    for key in ("pseudo_pre","pseudo_post"):
        out = tmp_path/(key+".png");Image.fromarray(value).save(out);row[key] = out.name
    path.write_text(json.dumps(data))
    ds = ChangeDataset(path,"train",augment=False)
    item = ds[0]
    assert set(item["pseudo_pre"].unique().tolist()) == {0,2}
    batch = collate_batch([item])
    logits = torch.zeros(1,3,16,16,requires_grad=True)
    out = {"pred_a":logits,"pred_b":logits.clone(),"pred_cd":torch.full((1,1,16,16),.5,requires_grad=True)}
    plain,_ = task_loss("scd",out,batch,pseudo_weight=0)
    augmented,parts = task_loss("scd",out,batch,pseudo_weight=1)
    assert float((augmented-plain).detach()) == pytest.approx(math.log(3),rel=1e-6)
    assert parts["pseudo"].requires_grad
    augmented.backward()
    assert logits.grad.isfinite().all()


def test_bda_normalization_matches_dense_encoder(tmp_path):
    path,_ = fixture_manifest(tmp_path,"bda",shapes=((8,8),))
    sample = ChangeDataset(path,"train",augment=False)[0]
    assert sample["pre"][0,0,0].item() == pytest.approx(127/127.5-1,abs=1e-7)
    assert sample["localization"].dtype == torch.long


def test_training_vocabulary_excludes_validation_tokens(tmp_path):
    path,data = fixture_manifest(tmp_path,"cc",shapes=((50,50),))
    data["splits"]["val"] = [{"id":"val","pre":"val_pre.png","post":"val_post.png","captions":[["heldoutonly"]]}]
    path.write_text(json.dumps(data))
    wordmap,meta = build_wordmap(path,min_word_freq=0)
    assert "heldoutonly" not in wordmap and meta["source"] == "train"
    dataset = ChangeDataset(path,"train",wordmap=wordmap,max_words=6)
    assert len(dataset) == 2
    item = dataset[0]
    assert item["pre"].shape == (3,256,256)
    assert item["original_size"] == (50,50)
    assert len(item["references"]) == 2
    values,length = encode_caption(["a","unseen"],wordmap,max_words=3)
    assert length == 4 and decode_caption(values,wordmap) == ["a","<unk>"]


def test_pixel_metrics_and_ignore_conventions():
    binary = BinaryMetrics();binary.update([[1,1],[0,0]],[[1,0],[1,0]])
    assert binary.compute()["F1"] == pytest.approx(.5)
    assert binary.compute()["IoU"] == pytest.approx(1/3)
    semantic = SemanticMetrics(3)
    y = np.array([[0,1],[2,0]])
    semantic.update(y,y,y,y)
    assert semantic.compute()["mIoU"] == 1
    assert semantic.compute()["Fscd"] == 1
    assert semantic.compute()["SeK"] == 1
    damage = DamageMetrics()
    target = np.array([[1,2,3,4,255]])
    loc = np.ones_like(target)
    damage.update(loc,np.array([[1,2,3,4,0]]),loc,target)
    assert damage.compute()["F1_overall"] == pytest.approx(1,abs=2e-7)
    # Predicted localization must not gate the classification metric.
    damage2 = DamageMetrics();damage2.update(np.zeros_like(loc),np.array([[1,2,3,4,0]]),loc,target)
    assert damage2.compute()["F1_cls"] == damage.compute()["F1_cls"]
    assert damage2.compute()["F1_loc"] == 0


def test_losses_ignore_padded_pixels_and_backpropagate():
    logits = torch.tensor([[[[-1.,2.],[10.,10.]]]],requires_grad=True)
    label = torch.tensor([[[[0.,1.],[0.,0.]]]])
    valid = torch.tensor([[[True,True],[False,False]]])
    loss,_ = task_loss("bcd",{"change_logits":logits},{"label":label,"valid_mask":valid})
    small,_ = task_loss("bcd",{"change_logits":logits[:,:,:1]},{"label":label[:,:,:1],"valid_mask":valid[:,:1]})
    assert torch.equal(loss,small)
    loss.backward()
    assert logits.grad[0,0,1].eq(0).all() and logits.grad[0,0,0].abs().sum() > 0
    cc = torch.randn(1,3,7,requires_grad=True)
    value,_ = task_loss("cc",{"caption_logits":cc},{"tokens":torch.tensor([[1,2,3,0]])})
    value.backward();assert torch.isfinite(cc.grad).all()


def test_token_caption_metrics_and_generated_update():
    words = ["a","new","house","appears","here"]
    assert caption_scores([[words]],[words])["Bleu_4"] == pytest.approx(1)
    assert caption_scores([[words]],[words])["ROUGE_L"] == pytest.approx(1)
    assert caption_scores([[words]],[[]])["Bleu_4"] == 0
    wordmap = {"<pad>":0,"<unk>":1,"<start>":2,"<end>":3,**{w:i+4 for i,w in enumerate(words)}}
    meter = create_metric("cc")
    update_metrics("cc",meter,{"generated_tokens":torch.tensor([[2,4,5,6,7,8,3,0]])},{"references":[[words]]},wordmap=wordmap)
    assert meter.compute()["ROUGE_L"] == 1


def test_pseudo_thresholds_match_sequential_classwise_accumulation():
    from statevid.pseudo import ClasswisePseudoLabeler,pseudo_loss_weight
    labeler = ClasswisePseudoLabeler(3)
    def probability(confidence,category=1):
        values = torch.full((1,3,2,2),(1-confidence)/2)
        values[:,category] = confidence
        return values
    gt = torch.zeros((1,2,2),dtype=torch.long)
    result = labeler.generate(probability(.8),probability(.7),gt,gt)
    assert result["diagnostics"]["thresholds_pre"][1] == pytest.approx((.6+.8*4)/5,abs=1e-7)
    assert result["diagnostics"]["thresholds_post"][1] == pytest.approx((.6+.8*4+.7*4)/9,abs=1e-7)
    assert not result["accepted"].any()
    restored = ClasswisePseudoLabeler(3);restored.load_state_dict(labeler.state_dict())
    assert np.array_equal(restored.thresholds,labeler.thresholds)
    assert pseudo_loss_weight(9) == 0
    assert pseudo_loss_weight(10) == pytest.approx(.04)
    assert pseudo_loss_weight(14) == pytest.approx(.2)


def test_pseudo_acceptance_respects_agreement_gt_and_validity():
    from statevid.pseudo import ClasswisePseudoLabeler
    confident = torch.tensor([.01,.98,.01])[None,:,None,None].expand(1,3,2,2)
    labeler = ClasswisePseudoLabeler(3)
    gt = torch.zeros((1,2,2),dtype=torch.long)
    other = confident.flip(1)
    # Flipping [low,high,low] is unchanged; use an explicit different class.
    other = torch.tensor([.01,.01,.98])[None,:,None,None].expand_as(confident)
    assert not labeler.generate(confident,other,gt,gt)["accepted"].any()
    labeler = ClasswisePseudoLabeler(3)
    valid = torch.tensor([[[True,False],[True,True]]])
    changed = gt.clone();changed[0,1,0] = 2
    result = labeler.generate(confident,confident,changed,changed,valid=valid)
    assert result["accepted"].tolist() == [[[True,False],[False,True]]]
    assert torch.equal(result["target_pre"][~result["accepted"]],changed[~result["accepted"]])
