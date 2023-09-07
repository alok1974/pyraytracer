import random
import threading
from itertools import product
from typing import List, Optional, Tuple

import numpy as np

from .camera import Camera
from .color import Color
from .hittable import HitData, Hittable
from .interactive_render import InteractiveRenderWindow
from .light import PointLight
from .render_image import RenderImage
from .render_settings import RenderSettings
from .scene import Scene
from .util import progress_bar_context, random_range
from .vec3 import Vec3


class Renderer:
    def __init__(self, scene: Scene, render_settings: RenderSettings) -> None:
        self._scene = scene
        self._settings = render_settings

        aa_samples = self._settings.aa_samples
        if aa_samples < 1.0:
            raise RuntimeError(f'aa_samples={aa_samples} cannot be less that 1.0')

        self._aa_samples: float = aa_samples

        self._image_width: int = int(self._settings.image_width * self._aa_samples)
        self._image_height: int = int(self._settings.image_height * self._aa_samples)
        self._bg_color: Color = self._settings.bg_color or Color(r=0, g=0, b=0)

        self._camera: Camera = self._scene.camera
        self._hittables: List[Hittable] = self._scene.hittables
        self._lights: List[PointLight] = self._scene.lights

        self._image = RenderImage(
            width=self._image_width,
            height=self._image_height,
            bg_color=self._bg_color,
        )

        self._random_circular_arr: List[Tuple[Vec3, float, Color]] = []
        self._random_tail_arr: List[Tuple[float, Vec3, float, Vec3, Vec3, Vec3, Vec3, Color, float]] = []

        if self._settings.interactive_render:
            self._ir = InteractiveRenderWindow(
                image=self._image.image,
                close_after_rendering=self._settings.close_interactive_render,
            )

    def save_image(self) -> None:
        output_path = self._settings.output_path
        self._image.save(output_path=output_path, antialias_samples=self._aa_samples)

    def run(self) -> None:
        if self._settings.interactive_render:
            # Start the rendering loop in a separate thread
            rendering_thread = threading.Thread(target=self._render)
            rendering_thread.daemon = True
            rendering_thread.start()

            # Start interactive render window
            self._ir.run()
        else:
            self._render()

    def _render(self) -> None:
        with progress_bar_context() as update_bar:
            counter = 1
            for y, x in product(range(self._image_height), range(self._image_width)):
                if self._camera is None:
                    continue

                pixel_pos = self._get_pixel_pos(x=x, y=y)
                ray_dir = Vec3.normalize(v=pixel_pos - self._camera.center)
                pixel_color = self._raytrace(ray_dir=ray_dir)

                # If ray did not hit an object, get a circle
                # with random color and size
                pixel_color = self._add_circular_noise(
                        pixel_color=pixel_color,
                        x=x,
                        y=y,
                )

                # Add tail noise (a circle with a tail)
                pixel_color = self._add_tail_noise(
                    pixel_color=pixel_color,
                    x=x,
                    y=y,
                )

                # Add pixel noise
                pixel_color = self._add_pixel_noise(
                    pixel_color=pixel_color,
                    x=x,
                    y=y,
                )

                if pixel_color is not None:
                    self._image.add_pixel_color(x=x, y=y, color=pixel_color)

                percentage = (counter / (self._image_width * self._image_height)) * 100
                update_bar(percentage=percentage)
                counter += 1

                if self._settings.interactive_render and percentage % 2 == 0:
                    self._ir.update_pixels(self._image.image)

        if self._settings.interactive_render:
            self._ir.on_rendering_complete()

    def _add_circular_noise(
            self, pixel_color: Optional[Color],
            x: int,
            y: int,
    ) -> Optional[Color]:

        if pixel_color is None and self._settings.add_circular_noise:
            return self._circular_noise(x=x, y=y)
        else:
            return pixel_color

    def _circular_noise(self, x: int, y: int) -> Optional[Color]:
        scale = self._settings.circular_noise_scale
        is_random_pixel = self._pick_random_pixel(x=x, y=y, scale=scale)
        not_border_pixel = self._is_not_border_pixel(x=x, y=y)

        if is_random_pixel and not_border_pixel:
            self._update_random_circular_arr(x=x, y=y)

        for center, radius, s_color in self._random_circular_arr:
            x_norm, y_norm = self._pixel_to_normalized(x=x, y=y)
            in_circle = self._point_in_circle(
                point=Vec3(x=x_norm, y=y_norm, z=0),
                center=center,
                radius=radius,
            )
            if in_circle:
                return s_color

        return None

    def _add_tail_noise(
            self, pixel_color: Optional[Color], x: int, y: int,
    ) -> Optional[Color]:

        if pixel_color is None and self._settings.add_tail_noise:
            return self._tail_noise(x=x, y=y)
        else:
            return pixel_color

    def _tail_noise(self, x: int, y: int) -> Optional[Color]:
        scale = self._settings.tail_noise_scale
        is_random_pixel = self._pick_random_pixel(x=x, y=y, scale=scale)
        not_border_pixel = self._is_not_border_pixel(x=x, y=y)

        if is_random_pixel and not_border_pixel:
            self._update_random_tail_arr(x=x, y=y)

        for s_x, c, r, q1, q2, q3, q4, s_color, angle in self._random_tail_arr:
            x_norm, y_norm = self._pixel_to_normalized(x=x, y=y)
            p = Vec3(x=x_norm, y=y_norm, z=0)
            in_circle = self._point_in_circle(point=p, center=c, radius=r)
            in_quad = self._point_in_quad(point=p, a=q1, b=q2, c=q3, d=q4)
            lerp_factor = self._fade_to_tail(
                x=x_norm,
                s_x=s_x,
                r=r,
                angle=angle,
                width=self._settings.tail_noise_width,
            )
            random_in = random.random() < lerp_factor

            if (in_circle or in_quad) and random_in:
                return lerp_factor * s_color + (1 - lerp_factor) * self._settings.bg_color

        return None

    def _fade_to_tail(self, x: float, s_x: float, r: float, width: float, angle: float) -> float:
        value = 1 - ((x - s_x) / (width * r))
        angle_rad = np.radians(angle)
        return value * np.cos(angle_rad)

    def _rotate_point(self, rot_center: Vec3, rot_degrees: float, to_rotate: Vec3) -> Vec3:
        to_rotate_arr = np.array([to_rotate.x, to_rotate.y])
        rot_center_arr = np.array([rot_center.x, rot_center.y])
        theta_rad = np.radians(rot_degrees)

        rotation_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)]
        ])

        to_rotate_translated = to_rotate_arr - rot_center_arr
        translated_rotated = rotation_matrix @ to_rotate_translated
        rotated = translated_rotated + rot_center_arr

        return Vec3(x=rotated[0], y=rotated[1], z=0)

    def _point_in_circle(self, point: Vec3, center: Vec3, radius: float) -> bool:
        return (point - center).length() <= radius

    def _point_in_quad(self, point: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> bool:
        def triangle_area(v1: Vec3, v2: Vec3, v3: Vec3) -> float:
            return 0.5 * abs(v1.x * (v2.y - v3.y) + v2.x * (v3.y - v1.y) + v3.x * (v1.y - v2.y))

        p = point
        quad_area = triangle_area(a, b, c) + triangle_area(a, c, d)
        sum_triangles_area = (
            triangle_area(p, a, b)
            + triangle_area(p, b, c)
            + triangle_area(p, c, d)
            + triangle_area(p, d, a)
        )

        return abs(quad_area - sum_triangles_area) < 1e-9  # Using a small threshold for floating point comparison

    def _add_pixel_noise(self, pixel_color: Optional[Color], x: int, y: int) -> Optional[Color]:
        if pixel_color is None and self._settings.add_pixel_noise:
            return self._pixel_noise(x=x, y=y)
        else:
            return pixel_color

    def _pixel_noise(self, x: int, y: int) -> Optional[Color]:
        scale = self._settings.pixel_noise_scale
        is_random_pixel = self._pick_random_pixel(x=x, y=y, scale=scale)
        not_border_pixel = self._is_not_border_pixel(x=x, y=y)

        if is_random_pixel and not_border_pixel:
            color_min = self._settings.pixel_noise_color_min
            color_max = self._settings.pixel_noise_color_max
            return self._random_color(color_min=color_min, color_max=color_max)

        return None

    def _pick_random_pixel(self, x: int, y: int, scale: float) -> bool:
        max_range_x = max(int(scale * self._image_width), 2)
        max_range_y = max(int(scale * self._image_height), 2)
        random_x = x % random.choice(range(1, max_range_x)) == 0
        random_y = y % random.choice(range(1, max_range_y)) == 0
        is_random_pixel = random_x and random_y

        return is_random_pixel

    def _is_not_border_pixel(self, x: int, y: int) -> bool:
        not_border_x = 0 < x < self._image_width - 1
        not_border_y = 0 < y < self._image_height - 1
        not_border_pixel = not_border_x and not_border_y

        return not_border_pixel

    def _update_random_circular_arr(self, x: int, y: int) -> None:
        color_min = self._settings.circular_noise_color_min
        color_max = self._settings.circular_noise_color_max
        r_color = self._random_color(color_min=color_min, color_max=color_max)

        size_min = self._settings.circular_noise_size_min
        size_max = self._settings.circular_noise_size_max
        r = random_range(size_min, size_max) / 2
        x_norm, y_norm = self._pixel_to_normalized(x=x, y=y)
        c = Vec3(x=x_norm + r, y=y_norm - r, z=0)
        self._random_circular_arr.append((c, r, r_color))

    def _update_random_tail_arr(self, x: int, y: int) -> None:
        color_min = self._settings.tail_noise_color_min
        color_max = self._settings.tail_noise_color_max
        r_color = self._random_color(color_min=color_min, color_max=color_max)

        size_min = self._settings.tail_noise_size_min
        size_max = self._settings.tail_noise_size_max
        relative_tail_width = self._settings.tail_noise_width

        x_norm, y_norm = self._pixel_to_normalized(x=x, y=y)

        d = random_range(size_min, size_max)
        r = d / 2
        absolute_tail_width = relative_tail_width * r
        c = Vec3(x=x_norm + r, y=y_norm - (3 * r), z=0)
        q1 = Vec3(x=x_norm + r, y=y_norm - d, z=0)
        q2 = Vec3(x=x_norm + absolute_tail_width, y=y_norm, z=0)
        q3 = Vec3(x=x_norm + absolute_tail_width, y=y_norm - (3 * d), z=0)
        q4 = Vec3(x=x_norm + r, y=y_norm - (2 * d), z=0)

        # Randomly rotate points
        angle_min = self._settings.tail_noise_angle_min
        angle_max = self._settings.tail_noise_angle_max
        angle = random_range(angle_min, angle_max)
        q1 = self._rotate_point(rot_center=c, rot_degrees=angle, to_rotate=q1)
        q2 = self._rotate_point(rot_center=c, rot_degrees=angle, to_rotate=q2)
        q3 = self._rotate_point(rot_center=c, rot_degrees=angle, to_rotate=q3)
        q4 = self._rotate_point(rot_center=c, rot_degrees=angle, to_rotate=q4)

        # Shift all points down below y_pos so that we can render them when
        # we hit x_pos, y_pos
        if q2.y > y_norm:
            delta = q2.y - y_norm
            c = Vec3(x=c.x, y=c.y - delta, z=0)
            q1 = Vec3(x=q1.x, y=q1.y - delta, z=0)
            q2 = Vec3(x=q2.x, y=q2.y - delta, z=0)
            q3 = Vec3(x=q3.x, y=q3.y - delta, z=0)
            q4 = Vec3(x=q4.x, y=q4.y - delta, z=0)

        self._random_tail_arr.append((x_norm, c, r, q1, q2, q3, q4, r_color, angle))

    def _random_color(self, color_min: int, color_max: int) -> Color:
        return Color(
            r=random.randint(color_min, color_max),
            g=random.randint(color_min, color_max),
            b=random.randint(color_min, color_max),
        )

    def _raytrace(self, ray_dir: Vec3) -> Optional[Color]:
        last_color = self._settings.bg_color
        sorted_hittables = self._hit_data_sorted_by_t(ray_dir=ray_dir)
        if sorted_hittables is None:
            return None

        if len(sorted_hittables) > 1:
            # Last before closest hittable and hit data
            l_hittable, l_hit_data = sorted_hittables[1]
            last_color = self._get_pixel_color(
                hittable=l_hittable,
                hit_data=l_hit_data,
                last_color=last_color,
            ) or last_color

        hittable, hit_data = sorted_hittables[0]
        return self._get_pixel_color(
            hittable=hittable,
            hit_data=hit_data,
            last_color=last_color,
        )

    def _hit_data_sorted_by_t(self, ray_dir: Vec3) -> Optional[List[Tuple[Hittable, HitData]]]:
        hittables_with_t = []
        for hittable in self._hittables:
            hit_data = hittable.get_hit_data(
                ray=ray_dir,
                ray_origin=self._camera.center,
            )
            if hit_data.t is not None:
                hittables_with_t.append((hit_data.t, hittable, hit_data))

        if not hittables_with_t:
            return None

        return [(h[1], h[2]) for h in sorted(hittables_with_t)]

    def _get_pixel_color(
            self, hittable: Hittable,
            hit_data: HitData,
            last_color: Color
    ) -> Optional[Color]:

        diffuse = hittable.material.color
        if hittable.material.is_constant:
            return diffuse

        if hittable.material.random_color_range:
            a, b = hittable.material.random_color_range
            diffuse = Color(
                r=int(diffuse.r * random_range(a, b)),
                g=int(diffuse.g * random_range(a, b)),
                b=int(diffuse.b * random_range(a, b)),
            )

        final_color = Color(r=0, g=0, b=0)
        for light in self._lights:
            # Check light linking
            if light.is_excluded(hittable=hittable):
                continue

            elif light.is_included(hittable=hittable):
                l_factor = self._lambertian_factor(
                    light=light,
                    hit_data=hit_data,
                )
                reflectance = diffuse * l_factor

                # Lerp of light and diffuse color with t = 0.5
                final_color += ((0.5 * reflectance) + (0.5 * light.color)) * light.intensity

        if hittable.material.transparency > 0.0:
            t_factor = hittable.material.transparency
            final_color = t_factor * last_color + (1 - t_factor) * final_color

        if hittable.material.dispersion > 0.0:
            if random.random() < hittable.material.dispersion:
                return last_color
            else:
                return final_color

        return final_color

    def _lambertian_factor(self, light: PointLight, hit_data: HitData) -> float:
        if hit_data.point is None or hit_data.normal is None:
            return 0

        to_light = Vec3.normalize(light.center - hit_data.point)

        # Lambertian Cosine model
        lambertian_factor = float(max(0, Vec3.dot(to_light, hit_data.normal)))

        return lambertian_factor

    def _get_pixel_pos(self, x: int, y: int) -> Vec3:
        px = (2 * (x + 0.5) / self._image_width - 1) * self._image.aspect * self._camera.fov
        py = (1 - 2 * (y + 0.5) / self._image_height) * self._camera.fov

        # Calculate the direction in camera space
        camera_dir = -1 * self._camera.right * px + self._camera.upward * py

        # The image plane is one unit away from the camera
        image_plane_pos = self._camera.center + self._camera.forward

        # Compute world space pixel position
        return image_plane_pos + camera_dir

    def _pixel_to_normalized(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to normalized screen space
        with aspect ratio adjustment.
        """
        x_normalized = (2 * (x / self._image.width) - 1) * self._image.aspect
        y_normalized = 1 - 2 * (y / self._image_height)

        return x_normalized, y_normalized
