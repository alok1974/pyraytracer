import json
from enum import Enum, unique
from pathlib import Path
from typing import List, Tuple, Union

from pydantic import ValidationError

from .color import Color
from .render_settings import RenderSettings


class SettingsValidationError(Exception):
    pass


@unique
class ObjectAttrs(Enum):
    BG_COLOR = 'bg_color'


class SettingsLoader:
    def __init__(self, setting_file_path: Union[str, Path]) -> None:
        self._settings_file_path = setting_file_path
        self._settings_data: dict = self._load_settings_file()

        self._errors: List[str] = []
        self._settings: RenderSettings = self._load_setting()

    @property
    def settings(self) -> RenderSettings:
        return self._settings

    def _load_settings_file(self) -> dict:
        try:
            with open(self._settings_file_path, 'r') as fp:
                return json.load(fp=fp)
        except json.decoder.JSONDecodeError as e:
            msg = f'Malformed settings file:\n{e}'
            raise SettingsValidationError(msg)

    def _load_setting(self) -> RenderSettings:
        settings_data, data_ok = self._update_object(
            object_data=self._settings_data
        )
        if data_ok:
            try:
                return RenderSettings(**settings_data)
            except ValidationError as e:
                raise SettingsValidationError(
                    f'Settings has following errors:\n{e}'
                )
        else:
            errors = '\n'.join(self._errors)
            raise SettingsValidationError(
                f'Settings has following error: {errors}'
            )

    def _update_object(self, object_data: dict) -> Tuple[dict, bool]:
        if ObjectAttrs.BG_COLOR.value not in object_data:
            return object_data, True

        color_data = object_data[ObjectAttrs.BG_COLOR.value]
        if (
            not isinstance(color_data, list)
            or len(color_data) != 3
            or not all([isinstance(t, int) for t in color_data])
        ):
            self._errors.append(
                f'The field "{ObjectAttrs.BG_COLOR.value}" of '
                f'{RenderSettings.__name__}'
                f'should be a list of 3 ints, got {color_data}'
            )
            return object_data, False
        try:
            r, g, b = color_data
            color = Color(r=r, g=g, b=b)
        except ValidationError as e:
            self._errors.append(
                f'{RenderSettings.__name__}'
                f'needs a field {ObjectAttrs.BG_COLOR.value}='
                f'{Color.__name__}({r}, {g}, {b}) but there were following '
                f' validation errors while creating an object_data '
                f'of {Color.__name__}: \n{e}'
            )
            return object_data, False
        else:
            object_data[ObjectAttrs.BG_COLOR.value] = color
            return object_data, True
