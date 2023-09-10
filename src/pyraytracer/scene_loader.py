import json
from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Union

from pydantic_core import ValidationError

from .camera import Camera
from .color import Color
from .cube import Cube
from .light import PointLight
from .material import Material
from .scene import Scene
from .sphere import Sphere
from .vec3 import Vec3


@unique
class SceneKeys(Enum):
    CAMERA = 'camera'
    MATERIALS = 'materials'
    SPHERES = 'spheres'
    CUBES = 'cubes'
    LIGHTS = 'lights'
    LIGHT_LINKING = 'light_linking'


@unique
class ObjectAttrs(Enum):
    NAME = 'name'
    MATERIAL = 'material'
    INCLUDE = 'include'
    EXCLUDE = 'exclude'
    COLOR = 'color'
    CENTER = 'center'
    INTENSITY = 'intensity'
    LOOK_AT = 'look_at'
    SCALE = 'scale'
    ROTATION = 'rotation'


class SceneValidationError(Exception):
    pass


class SceneLoader:
    def __init__(self, scene_file_path: Union[Path, str]) -> None:
        self._scene_file_path = scene_file_path

        self._scene_data: dict = self._load_scene_file()

        self._materials: Dict[str, Material] = {}
        self._spheres: Dict[str, Sphere] = {}
        self._cubes: Dict[str, Cube] = {}
        self._lights: Dict[str, PointLight] = {}

        self._errors: List[str] = []

        self._scene: Scene = Scene()
        self._load_scene()

    @property
    def scene(self) -> Scene:
        return self._scene

    def _load_scene_file(self) -> dict:
        try:
            with open(self._scene_file_path, 'r') as fp:
                return json.load(fp=fp)
        except json.decoder.JSONDecodeError as e:
            msg = f'Malformed scene file:\n{e}'
            raise SceneValidationError(msg)

    def _load_scene(self) -> None:
        self._check_root_keys()
        if self._errors:
            raise SceneValidationError('\n'.join(self._errors))

        self._load_camera()
        self._load_materials()
        self._load_spheres()
        self._load_cubes()
        self._load_lights()
        self._load_light_linking()

        if self._errors:
            raise SceneValidationError('\n'.join(self._errors))

        return None

    def _load_camera(self) -> None:
        camera_data = self._scene_data[SceneKeys.CAMERA.value]
        camera_data, data_ok = self._update_object_data(
            object_data=camera_data,
            object_cls=Camera
        )

        if data_ok:
            try:
                self._scene.camera = Camera(**camera_data)
            except ValidationError as e:
                camera_type = Camera.__name__
                camera_name = camera_data[ObjectAttrs.NAME.value]
                self._errors.append(
                    f'{camera_type} "{camera_name}" has the following '
                    f'errors for params/value: \n{e}'
                )

        return None

    def _load_materials(self) -> None:
        if SceneKeys.MATERIALS.value not in self._scene_data:
            return

        self._load_data_list(scene_key=SceneKeys.MATERIALS)

    def _load_spheres(self) -> None:
        if SceneKeys.SPHERES.value not in self._scene_data:
            return

        self._load_data_list(scene_key=SceneKeys.SPHERES)
        self._scene.add_hittables(hittables=list(self._spheres.values()))

    def _load_cubes(self) -> None:
        if SceneKeys.CUBES.value not in self._scene_data:
            return

        self._load_data_list(scene_key=SceneKeys.CUBES)
        self._scene.add_hittables(hittables=list(self._cubes.values()))

    def _load_lights(self) -> None:
        if SceneKeys.LIGHTS.value not in self._scene_data:
            return
        self._load_data_list(scene_key=SceneKeys.LIGHTS)
        self._scene.add_lights(lights=list(self._lights.values()))

    def _load_data_list(self, scene_key: SceneKeys) -> None:
        data_list: Dict[str, Any] = {}
        object_cls: Any = None

        if scene_key == SceneKeys.MATERIALS:
            data_list = self._materials
            object_cls = Material
        elif scene_key == SceneKeys.SPHERES:
            data_list = self._spheres
            object_cls = Sphere
        elif scene_key == SceneKeys.CUBES:
            data_list = self._cubes
            object_cls = Cube
        elif scene_key == SceneKeys.LIGHTS:
            data_list = self._lights
            object_cls = PointLight
        else:
            raise SceneValidationError(f"Unknown scene key {scene_key}")

        for object_data in self._scene_data.get(scene_key.value, []):
            object_data, data_ok = self._update_object_data(object_data=object_data, object_cls=object_cls)
            if data_ok:
                object_name = object_data[ObjectAttrs.NAME.value]
                try:
                    data_list[object_name] = object_cls(**object_data)
                except ValidationError as e:
                    self._errors.append(
                        f'{object_cls.__name__} "{object_name}" has '
                        f'the following errors for param/value: \n{e}'
                    )

    def _load_light_linking(self) -> None:
        if SceneKeys.LIGHT_LINKING.value not in self._scene_data:
            return

        self._check_light_linking()
        if self._errors:
            return

        include_attr = ObjectAttrs.INCLUDE.value
        exclude_attr = ObjectAttrs.EXCLUDE.value

        for link_data in self._scene_data[SceneKeys.LIGHT_LINKING.value]:
            light_name = link_data[ObjectAttrs.NAME.value]
            light = self._lights[light_name]

            # Add includes
            self._add_includes_excludes(
                link_data=link_data,
                include_exclude_attr=include_attr,
                light_name=light_name,
                light=light,
            )

            # Add excludes
            self._add_includes_excludes(
                link_data=link_data,
                include_exclude_attr=exclude_attr,
                light_name=light_name,
                light=light,
            )

    def _add_includes_excludes(
            self, link_data: dict, include_exclude_attr: str,
            light_name: str, light: PointLight
    ) -> None:

        if include_exclude_attr not in link_data:
            return None

        include_exclude_func = (
            light.add_include
            if include_exclude_attr == ObjectAttrs.INCLUDE.value
            else light.add_exclude
        )

        hittable: Any = None
        for hittable_name in link_data[include_exclude_attr]:
            hittable = self._spheres.get(hittable_name) or self._cubes.get(hittable_name)
            if hittable is None:
                self._errors.append(
                    f'Light "{light_name}" has a hittable "{hittable_name}" '
                    f'defined in the {include_exclude_attr}, but this hittable '
                    f'was not loaded in the scene.'
                )
            else:
                include_exclude_func(hittable=hittable)

        return None

    def _check_root_keys(self) -> None:
        scene_data_keys = sorted(self._scene_data.keys())
        expected_keys = sorted(k.value for k in SceneKeys)

        unknown_keys = [k for k in scene_data_keys if k not in expected_keys]
        unknown_str = f'some unknown keys: {unknown_keys} '

        required_keys = [SceneKeys.CAMERA.value]
        missing_keys = [k for k in required_keys if k not in scene_data_keys]
        missing_str = f'some missing keys: {missing_keys}'

        msg = 'Scene data has '
        if all([unknown_keys, missing_keys]):
            msg += f'{missing_str} and {unknown_str}'
            self._errors.append(msg)
            return None
        elif missing_keys:
            msg += missing_str
            self._errors.append(msg)
            return None
        elif unknown_keys:
            msg += unknown_str
            self._errors.append(msg)
            return None

        for key, value in self._scene_data.items():
            if key == SceneKeys.CAMERA.value:
                if not isinstance(value, dict):
                    self._errors.append(
                        f'Scene root key "{key}": should '
                        f'be a dict, got {type(value)}'
                    )
            else:
                if not isinstance(value, list):
                    self._errors.append(
                        f'Scene root key "{key}" should '
                        f'be a list, got {type(value)}'
                    )

    def _check_light_linking(self) -> None:
        defined_light_names = self._get_scene_object_names(
            scene_key=SceneKeys.LIGHTS.value,
        )

        expected_attrs = [
            ObjectAttrs.NAME.value,
            ObjectAttrs.INCLUDE.value,
            ObjectAttrs.EXCLUDE.value,
        ]

        light_linking_data = self._scene_data.get(SceneKeys.LIGHT_LINKING.value, [])
        for link_data in light_linking_data:
            for attr in link_data.keys():
                if attr not in expected_attrs:
                    self._errors.append(
                        f'Unknon attr {attr} in light linking data {link_data}'
                        f', expected attrs {expected_attrs}'
                    )
            if ObjectAttrs.NAME.value not in link_data:
                self._errors.append(
                    f'"name" attr missing from link data {link_data}'
                )
            else:
                light_name = link_data[ObjectAttrs.NAME.value]
                if light_name not in defined_light_names:
                    self._errors.append(
                        f'light={light_name} used in light_linking '
                        'is not defined in the scene.'
                    )

                self._check_includes_excludes(
                    link_data=link_data,
                    include_exclude_key=ObjectAttrs.INCLUDE.value,
                )

                self._check_includes_excludes(
                    link_data=link_data,
                    include_exclude_key=ObjectAttrs.EXCLUDE.value,
                )

    def _check_includes_excludes(
            self,
            link_data: dict,
            include_exclude_key: str,
    ) -> None:

        if include_exclude_key not in link_data:
            return None

        include_exclude_data = link_data[include_exclude_key]
        is_data_a_list = isinstance(include_exclude_data, list)
        is_data_list_of_strings = all(
            [
                isinstance(t, str)
                for t in include_exclude_data
            ],
        )
        if not is_data_a_list or not is_data_list_of_strings:
            self._errors.append(
                f'light link {include_exclude_key} '
                f'data={include_exclude_data} '
                f'should be a list of str'
            )

        light_name = link_data[ObjectAttrs.NAME.value]
        hittable_names = []
        sphere_names = self._get_scene_object_names(
            scene_key=SceneKeys.SPHERES.value,
        )
        hittable_names.extend(sphere_names)

        cube_names = self._get_scene_object_names(
            scene_key=SceneKeys.CUBES.value
        )
        hittable_names.extend(cube_names)
        for hittable in include_exclude_data:
            if hittable not in hittable_names:
                self._errors.append(
                    f'Hittable={hittable} defined in {include_exclude_key} '
                    f'of light {light_name} is not defined in scene'
                )

    def _update_object_data(
            self, object_data: dict,
            object_cls: Type,
    ) -> Tuple[dict, bool]:

        if not isinstance(object_data, dict):
            self._errors.append(
                f'object_data={object_data} should be of type dict'
            )
            return object_data, False

        name_attr = ObjectAttrs.NAME.value
        if name_attr not in object_data:
            self._errors.append(
                f'The "{name_attr}" attr is either missing or misspelt on '
                f'"{object_cls.__name__}" with following '
                f'data:\n{json.dumps(object_data, indent=4)}')
            return object_data, False

        object_data, center_updated = self._update_vector_data(
            attr_key=ObjectAttrs.CENTER.value,
            object_data=object_data,
            object_cls=object_cls,
        )

        object_data, look_at_updated = self._update_vector_data(
            attr_key=ObjectAttrs.LOOK_AT.value,
            object_data=object_data,
            object_cls=object_cls,
        )

        object_data, scale_updated = self._update_vector_data(
            attr_key=ObjectAttrs.SCALE.value,
            object_data=object_data,
            object_cls=object_cls,
        )

        object_data, rotation_updated = self._update_vector_data(
            attr_key=ObjectAttrs.ROTATION.value,
            object_data=object_data,
            object_cls=object_cls,
        )

        object_data, color_updated = self._update_color_data(
            object_data=object_data,
            object_cls=object_cls,
        )

        object_data, material_updated = self._update_material_data(
            object_data=object_data,
            object_cls=object_cls,
        )

        all_updated = all(
            [
                center_updated,
                scale_updated,
                rotation_updated,
                look_at_updated,
                color_updated,
                material_updated
            ]
        )

        return object_data, all_updated

    def _update_vector_data(
            self,
            attr_key: str,
            object_data: dict,
            object_cls: Type,
    ) -> Tuple[Dict, bool]:
        if attr_key not in object_data:
            return object_data, True

        attr_data = object_data[attr_key]
        if (
            not isinstance(attr_data, list)
            or len(attr_data) != 3
            or not all([isinstance(t, (float, int)) for t in attr_data])
        ):
            self._errors.append(
                f'The field "{attr_key}" of '
                f'{object_cls.__name__} "{object_data[ObjectAttrs.NAME.value]}" '
                f'should be a list of 3 numbers, got {attr_data}'
            )
            return object_data, False

        try:
            x, y, z = attr_data
            center = Vec3(x=x, y=y, z=z)
        except ValidationError as e:
            object_type = object_cls.__name__
            object_name = object_data[ObjectAttrs.NAME.value]
            self._errors.append(
                f'{object_type} "{object_name}" '
                f'needs a field {attr_key}='
                f'{Vec3.__name__}({x}, {y}, {z}) but there were following '
                f' validation errors while creating an object_data '
                f'of {Vec3.__name__}: \n{e}'
            )
            return object_data, False
        else:
            object_data[attr_key] = center
            return object_data, True

    def _update_color_data(
            self,
            object_data: dict,
            object_cls: Type,
    ) -> Tuple[Dict, bool]:

        if ObjectAttrs.COLOR.value not in object_data:
            return object_data, True

        color_data = object_data[ObjectAttrs.COLOR.value]
        if (
            not isinstance(color_data, list)
            or len(color_data) != 3
            or not all([isinstance(t, int) for t in color_data])
        ):
            object_type = object_cls.__name__
            object_name = object_data[ObjectAttrs.NAME.value]
            self._errors.append(
                f'The field "{ObjectAttrs.COLOR.value}" of '
                f'{object_type} "{object_name}" '
                f'should be a list of 3 ints, got {color_data}'
            )
            return object_data, False
        try:
            r, g, b = color_data
            color = Color(r=r, g=g, b=b)
        except ValidationError as e:
            object_type = object_cls.__name__
            object_name = object_data[ObjectAttrs.NAME.value]
            self._errors.append(
                f'{object_type} "{object_name}" '
                f'needs a field {ObjectAttrs.COLOR.value}='
                f'{Color.__name__}({r}, {g}, {b}) but there were following '
                f' validation errors while creating an object_data '
                f'of {Color.__name__}: \n{e}'
            )
            return object_data, False
        else:
            object_data[ObjectAttrs.COLOR.value] = color
            return object_data, True

    def _update_material_data(
            self,
            object_data: dict,
            object_cls: Type,
    ) -> Tuple[Dict, bool]:

        if ObjectAttrs.MATERIAL.value not in object_data:
            return object_data, True

        material_name = object_data[ObjectAttrs.MATERIAL.value]
        material = self._materials.get(material_name)
        if material is None:
            object_type = object_cls.__name__
            object_name = object_data[ObjectAttrs.NAME.value]
            self._errors.append(
                f'{object_type} "{object_name}" '
                f'needs a field material="{material_name}" but the material '
                'is not defined in the scene.'
            )
            return object_data, False
        object_data[ObjectAttrs.MATERIAL.value] = material
        return object_data, True

    def _get_scene_object_names(self, scene_key: str) -> List[str]:
        return [
            object_data[ObjectAttrs.NAME.value]
            for object_data in self._scene_data.get(scene_key, [])
            if ObjectAttrs.NAME.value in object_data
        ]
