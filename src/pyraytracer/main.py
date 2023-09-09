from __future__ import annotations

from pathlib import Path
from typing import Union

from .render_settings import RenderSettings
from .renderer import Renderer
from .scene import Scene
from .scene_loader import SceneLoader
from .settings_loader import SettingsLoader


def main(
        scene_file_path: Union[str, Path],
        settings_file_path: Union[str, Path]
) -> None:

    scene: Scene = SceneLoader(scene_file_path=scene_file_path).scene
    settings: RenderSettings = SettingsLoader(setting_file_path=settings_file_path).settings

    renderer = Renderer(scene=scene, render_settings=settings)
    renderer.run()
    renderer.save_image()
