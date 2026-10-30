"""Rasp flags on top of the sn17gen generator: pre-refine candidates, photo tilt, keep-pose (stubbed)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sn17gen import config, jsutil, pipeline, prompts  # noqa: E402

JS = "export default function generate(THREE) { return new THREE.Group(); }"


def png(w, h, box, bg, fg):
    im = Image.new("RGB", (w, h), (bg,) * 3); ImageDraw.Draw(im).ellipse(box, fill=(fg,) * 3)
    out = io.BytesIO(); im.save(out, format="PNG"); return out.getvalue()


PHOTO_FACE = png(400, 400, (50, 80, 350, 330), 210, 90)
EDGE = png(518, 518, (60, 240, 460, 280), 255, 40)
FACE = png(518, 518, (80, 60, 440, 420), 255, 40)


class Resp:
    def __init__(self, text):
        self.output_text = text; self.usage = {"input_tokens": 10, "output_tokens": 5}


class Calls:
    """Answers generation/refine with distinct modules; the panel answers with penalties."""

    def __init__(self, panel=None):
        self.n = 0; self.log = []; self.panel = panel

    def create(self, *, request_id=0, **params):
        self.n += 1; self.log.append((request_id, params["model"], params["instructions"][:40]))
        if params["instructions"].startswith("You are a specialized 3D model evaluation system"):
            return Resp(json.dumps({"penalties": self.panel or [0] * 8}))
        return Resp(f"```js\n{JS} // v{self.n}\n```")


class Renderer:
    def __init__(self, front=FACE):
        self.front, self.log = front, []

    def render(self, source, options):
        self.log.append(source)
        thetas = getattr(options, "thetas", None) or options.get("thetas")
        single = len(thetas) == 1
        if single:
            return self.front if "Math.PI" not in source else FACE
        return png(1024, 1024, (100, 100, 900, 900), 255, 40)


def run(monkeypatch, *, keep_gen, tilt, keep_pose, renderer=None, panel=None, judge_refine=False, colour=False, astra=False):
    monkeypatch.setattr(config, "KEEP_GEN", keep_gen)
    monkeypatch.setattr(config, "PHOTO_TILT", tilt)
    monkeypatch.setattr(config, "KEEP_POSE", keep_pose)
    monkeypatch.setattr(config, "JUDGE_REFINE", judge_refine)
    monkeypatch.setattr(config, "COLOUR_RULES", colour)
    monkeypatch.setattr(config, "ASTRA_GEN", astra)
    calls = Calls(panel); r = renderer or Renderer()
    out = pipeline.run(PHOTO_FACE, 1, SimpleNamespace(responses=calls), r)
    return out, calls, r


def test_flags_off_reproduce_default_behaviour(monkeypatch):
    out, calls, r = run(monkeypatch, keep_gen=False, tilt=False, keep_pose=False)
    assert out is not None and "size0.y < 0.5" in out  # default final() heuristic wrapper
    assert all("Keep the pose" not in t for _, _, t in calls.log)


def test_keep_gen_adds_pre_refine_candidates_to_the_panel(monkeypatch):
    out, calls, _ = run(monkeypatch, keep_gen=True, tilt=False, keep_pose=False, panel=[9, 0, 9, 9])
    panel = [c for c in calls.log if c[1] == "gpt-6-astra"]
    assert len(panel) == 1 and out is not None
    assert "// v" in out  # a concrete module was chosen (second candidate per the penalties)


def test_keep_pose_changes_the_refine_prompt(monkeypatch):
    _, calls, _ = run(monkeypatch, keep_gen=False, tilt=False, keep_pose=True)
    assert "REFINE_USER_KEEP_POSE" in dir(prompts)
    assert calls.n >= 3


def test_photo_tilt_applies_when_the_front_view_is_edge_on(monkeypatch):
    out, _, r = run(monkeypatch, keep_gen=False, tilt=True, keep_pose=False, renderer=Renderer(front=EDGE))
    assert out is not None and "* Math.PI, 0, 0)" in out and "size0.y" not in out


def test_photo_tilt_keeps_face_on_modules(monkeypatch):
    out, _, _ = run(monkeypatch, keep_gen=False, tilt=True, keep_pose=False, renderer=Renderer(front=FACE))
    assert out is not None and "size0.y < 0.5" in out


def test_deterministic_across_runs(monkeypatch):
    a = run(monkeypatch, keep_gen=True, tilt=True, keep_pose=True, renderer=Renderer(front=EDGE))[0]
    b = run(monkeypatch, keep_gen=True, tilt=True, keep_pose=True, renderer=Renderer(front=EDGE))[0]
    assert a == b


def test_judge_refine_and_colour_rules_change_prompts(monkeypatch):
    class Spy(Calls):
        def __init__(self): super().__init__(); self.texts = []
        def create(self, *, request_id=0, **params):
            self.texts.append((params["instructions"], params["input"][0]["content"][0]["text"]))
            return super().create(request_id=request_id, **params)
    monkeypatch.setattr(config, "KEEP_GEN", False); monkeypatch.setattr(config, "PHOTO_TILT", False)
    monkeypatch.setattr(config, "KEEP_POSE", False); monkeypatch.setattr(config, "JUDGE_REFINE", True)
    monkeypatch.setattr(config, "COLOUR_RULES", True); monkeypatch.setattr(config, "ASTRA_GEN", False)
    spy = Spy(); out = pipeline.run(PHOTO_FACE, 1, SimpleNamespace(responses=spy), Renderer())
    assert out is not None
    assert any("#e6e6e6" in ins for ins, _ in spy.texts)
    assert any("REFINEMENT STEP" in txt and "Keep the pose" in txt for _, txt in spy.texts)


def test_astra_gen_uses_astra_high_for_generation(monkeypatch):
    out, calls, _ = run(monkeypatch, keep_gen=False, tilt=False, keep_pose=False, astra=True)
    gens = [c for c in calls.log if c[1] == "gpt-6-astra" and not c[2].startswith("You are a specialized")]
    assert out is not None and len(gens) >= 2  # 2 chains × gen on astra
