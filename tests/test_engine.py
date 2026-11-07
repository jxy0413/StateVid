"""Run the complete four-task loop on generated temporary image pairs."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from statevid.engine import train, load_checkpoint, evaluate_model
from statevid.data import ChangeDataset
from statevid.cli import main


@pytest.mark.parametrize("task", ["bcd","scd","bda","cc"])
def test_train_evaluate_predict_and_resume(tmp_path, task, monkeypatch):
    torch.set_num_threads(2)
    project = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    subprocess.run([sys.executable,str(project / "scripts/prepare_demo.py"),"--task",task,
                    "--output",str(data),"--size","32","--train-samples","2"],check=True,capture_output=True)
    manifest = data / "manifest.json"
    model = {"depth_factor":3}
    if task == "scd":
        model["num_classes"] = 3
    config = {"task":task,"manifest":str(manifest),"model":model,
              "train":{"max_steps":1,"workers":0,"batch_size":2,"amp":False,
                       "crop_size":32,"eval_every":1,"epochs":1},
              "caption":{"min_word_freq":0,"max_words":12,"image_size":32,"beam_size":1}}
    output = tmp_path / "run"
    result = train(config,output,device="cpu")
    assert result["state"] == "completed"
    model,saved = load_checkpoint(output / "best.pt")
    dataset = ChangeDataset(manifest,"test",augment=False,wordmap=saved.get("wordmap"),
                              max_words=12,cc_image_size=32)
    metrics = evaluate_model(model,dataset,device=torch.device("cpu"),wordmap=saved.get("wordmap"),beam_size=1)
    assert metrics["samples"] == 2 and metrics["loss"] >= 0
    destination = tmp_path / "predictions"
    monkeypatch.setattr(sys,"argv",["statevid","predict","--checkpoint",str(output / "best.pt"),
                                   "--manifest",str(manifest),"--output",str(destination),
                                   "--limit","1","--device","cpu","--beam-size","1"])
    main()
    assert len(list(destination.iterdir())) == 1
    assert any(p.suffix in (".png",".json") for p in destination.rglob("*"))
    resumed = train(config,output,device="cpu",max_steps=2,resume=output / "last.pt")
    assert resumed["step"] == 2
