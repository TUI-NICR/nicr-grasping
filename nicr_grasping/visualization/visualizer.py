import matplotlib.pyplot as plt

import pandas as pd

import open3d as o3d
import open3d.visualization.rendering as rendering
import open3d.visualization.gui as gui

from typing import Dict, Union

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nicr_grasping.datatypes.objects import Scene
    from nicr_grasping.datatypes.grasp.grasp_3d import Grasp3D
    from nicr_grasping.datatypes.grasp.grasp_lists import Grasp3DList
    from nicr_grasping.evaluation import EvalResults


def _get_grasp_info_string(grasp: 'Grasp3D', grasp_eval: Union[pd.DataFrame, None] = None) -> str:
    base_str = f'Confidence: {grasp.quality:.2f} ObjectID: {grasp.object_id}'
    if grasp_eval is not None:
        # need explicit string conversion as we fill to 5 characters
        # and otherwise bool would be printed as 0/1
        base_str += f' Collision: {str(grasp_eval.collision):<5}'
        base_str += f' Score: {1.2 - grasp_eval.min_friction:.2f}'
    return base_str


class GraspEvalVisualizer:

    object_cmap = plt.get_cmap('Set3')
    grasp_cmap = plt.get_cmap('jet')
    control_panel_width_scaling = 25
    control_panel_width = 400

    GRASP_COLOR_OPTIONS = ['Quality', 'Collision', 'ObjectID', 'Score']

    def __init__(self, scene: 'Scene') -> None:

        self._show_simple_grasps = False

        self._objects = {}
        self._grasps: Dict[str, 'Grasp3D'] = {}

        self.window = gui.Application.instance.create_window(
            "Grasp Evaluation Results", 1600, 1000)
        em = self.window.theme.font_size

        self.control_panel_width = self.control_panel_width_scaling * em

        self.window.set_on_key(self._on_key)

        layout = gui.Horiz()

        # create 3d scene
        self.scene_3d = gui.SceneWidget()
        self.scene_3d.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_3d.scene.set_background([1, 1, 1, 1])
        self.scene_3d.scene.scene.set_sun_light(
            [-1, -1, -1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self.scene_3d.scene.scene.enable_sun_light(True)
        bbox = o3d.geometry.AxisAlignedBoundingBox([-0.2, -0.2, -0.2],
                                                   [0.2, 0.2, 0.2])
        self.scene_3d.setup_camera(60, bbox, [0, 0, 0])

        self.scene_3d.frame = gui.Rect(self.window.content_rect.x, self.window.content_rect.y,
                                       self.window.content_rect.width - self.control_panel_width, self.window.content_rect.height)

        # layout.add_child(self.scene_3d)

        # add control panel
        tabs = gui.TabControl()
        control_panel = gui.Vert(em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
                                                 0.5 * em))

        toggle_buttons = gui.VGrid(1)
        # toggle_buttons.add_child(gui.Label("Show simple grasps"))
        toggle_grasp_collision_model = gui.ToggleSwitch('Show grasp collision models')
        toggle_grasp_collision_model.set_on_clicked(self._on_toggle_grasp_collision_model)
        self._toggle_grasp_collision_model = toggle_grasp_collision_model
        # toggle_simple_grasps_button.set_items(['Full', 'Simple'])

        toggle_buttons.add_child(toggle_grasp_collision_model)

        toggle_collision_filter = gui.ToggleSwitch('Filter collisions')
        toggle_collision_filter.set_on_clicked(self._on_toggle_collsion_filter)
        self._toggle_collision_filter = toggle_collision_filter

        toggle_buttons.add_child(toggle_collision_filter)

        control_panel.add_child(toggle_buttons)

        color_options = gui.Horiz()
        color_options.add_child(gui.Label("Grasp Color"))
        self.color_options_selection = gui.RadioButton(gui.RadioButton.VERT)
        self.color_options_selection.set_items(self.GRASP_COLOR_OPTIONS)
        self.color_options_selection.set_on_selection_changed(self._on_grasp_color_select)
        color_options.add_child(self.color_options_selection)

        control_panel.add_child(color_options)

        self.object_filter = gui.Combobox()
        self.object_filter.add_item("All")
        self.object_filter.set_on_selection_changed(self._on_object_filter)
        control_panel.add_child(self.object_filter)

        tabs.add_tab('Vis', control_panel)

        tabs.frame = gui.Rect(self.window.content_rect.width - self.control_panel_width, self.window.content_rect.y,
                              self.control_panel_width, self.window.content_rect.height)

        # layout.add_child(control_panel)

        # ADD GRASP INFO
        self.grasp_info_tab = gui.ListView()
        self.grasp_info_tab.set_on_selection_changed(self._on_grasp_list_selection)

        tabs.add_tab('Grasp Info', self.grasp_info_tab)

        self.window.add_child(self.scene_3d)
        self.window.add_child(tabs)

        # add scene to 3d view
        for oi, obj in enumerate(scene._objects):
            mat = rendering.MaterialRecord()
            obj_color = self.object_cmap(oi)
            mat.base_color = [
                *obj_color
            ]
            mat.shader = "defaultLit"

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(obj.mesh.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(obj.mesh.triangles)
            mesh.compute_vertex_normals()

            mesh = mesh.transform(obj.pose.transformation_matrix)

            obj_id = f'collision_object_{oi}'

            self.scene_3d.scene.add_geometry(
                obj_id,
                mesh,
                mat
            )

            self._objects[obj_id] = obj

    def _filter_grasp(self, grasp_index: int) -> bool:
        if self._toggle_collision_filter.is_on:
            grasp_eval = self._eval_results.get_info_for_grasp(grasp_index)
            if grasp_eval.collision:
                return False

        if self.object_filter.selected_text != "All":
            grasp = self._grasps[f'grasp_{grasp_index}']
            if grasp.object_id != int(self.object_filter.selected_text):
                return False

        return True

    def _on_toggle_collsion_filter(self, hide_collisions: bool) -> None:
        grasp_infos = []
        for gi, grasp in enumerate(self._grasps.values()):
            grasp_id = f'grasp_{gi}'
            if self._filter_grasp(gi):

                grasp_info = self._eval_results.get_info_for_grasp(gi)
                hide_grasp = grasp_info.collision and hide_collisions

                self.scene_3d.scene.show_geometry(grasp_id, not hide_grasp)

                if not hide_grasp:
                    grasp_infos.append(_get_grasp_info_string(grasp,
                                                              grasp_info))
            else:
                self.scene_3d.scene.show_geometry(grasp_id, False)

        self.grasp_info_tab.set_items(grasp_infos)

    def _on_object_filter(self, new_val: str, new_index: int) -> None:

        grasp_infos = []

        for gi, grasp in enumerate(self._grasps.values()):
            grasp_id = f'grasp_{gi}'

            show_grasp = False
            if new_val == "All":
                show_grasp = True
            elif grasp.object_id == int(new_val):
                show_grasp = True

            if show_grasp and self._filter_grasp(gi):
                grasp_eval = self._eval_results.get_info_for_grasp(gi)

                self.scene_3d.scene.show_geometry(grasp_id, True)
                grasp_infos.append(_get_grasp_info_string(grasp,
                                                          grasp_eval))
            else:
                self.scene_3d.scene.show_geometry(grasp_id, False)

        self.grasp_info_tab.set_items(grasp_infos)

    def _get_grasp_material(self, grasp_index: int) -> rendering.MaterialRecord:
        color_selection = self.color_options_selection.selected_value
        grasp_id = f'grasp_{grasp_index}'
        grasp = self._grasps[grasp_id]

        mat = rendering.MaterialRecord()

        if color_selection == 'Quality':
            mat.base_color = [
                *self.grasp_cmap(grasp.quality)
            ]
        elif color_selection == 'Collision':
            grasp_eval = self._eval_results.get_info_for_grasp(grasp_index)
            if grasp_eval.collision:
                mat.base_color = [1, 0, 0, 1]
            else:
                mat.base_color = [0, 1, 0, 1]
        elif color_selection == 'ObjectID' and grasp.object_id is not None:
            mat.base_color = [
                *self.object_cmap(float(grasp.object_id))
            ]
        elif color_selection == 'Score':
            grasp_eval = self._eval_results.get_info_for_grasp(grasp_index)
            mat.base_color = [
                *self.grasp_cmap(1.2 - grasp_eval.min_friction)
            ]

        mat.shader = "defaultLit"

        return mat

    def _on_grasp_color_select(self, idx: int) -> None:
        grasp_id = f'grasp_{idx}'
        for gi in range(len(self._grasps)):
            grasp_id = f'grasp_{gi}'

            mat = self._get_grasp_material(gi)
            self.scene_3d.scene.modify_geometry_material(grasp_id, mat)

    def _on_key(self, key: gui.KeyEvent) -> None:
        if key.key == ord('q') and key.type == gui.KeyEvent.UP:
            self.window.close()

    def _on_toggle_grasp_collision_model(self, show_grasp_collision_model: bool) -> None:
        # first we remote all graps
        for grasp_id in self._grasps:
            self.scene_3d.scene.remove_geometry(grasp_id)

        # then we add them again
        for gi, grasp in enumerate(self._grasps.values()):
            grasp_id = f'grasp_{gi}'

            g_mesh = grasp.open3d_geometry(simple_grasps=not show_grasp_collision_model)
            g_mesh.transform(grasp.transformation_matrix)

            mat = self._get_grasp_material(gi)

            self.scene_3d.scene.add_geometry(
                grasp_id,
                g_mesh,
                mat
            )

            self.scene_3d.scene.show_geometry(grasp_id, self._filter_grasp(gi))

    def _on_grasp_list_selection(self, new_val: str, is_double_click: bool) -> None:
        print(new_val, is_double_click)

    def _on_menu_quit(self) -> None:
        gui.Application.instance.quit()

    def add_grasps(self, grasps: 'Grasp3DList') -> None:
        grasp_info = []
        for gi, grasp in enumerate(grasps):
            g_mesh = grasp.open3d_geometry(simple_grasps=True)
            g_mesh.transform(grasp.transformation_matrix)

            mat = rendering.MaterialRecord()
            grasp_color = self.grasp_cmap(grasp.quality)
            mat.base_color = [
                *grasp_color
            ]
            mat.shader = "defaultLit"

            grasp_id = f'grasp_{gi}'

            self.scene_3d.scene.add_geometry(
                grasp_id,
                g_mesh,
                mat
            )

            self._grasps[grasp_id] = grasp

            grasp_info.append(_get_grasp_info_string(grasp))
        self.grasp_info_tab.set_items(grasp_info)

        unique_obj_ids = set([g.object_id for g in grasps])
        for obj_id in unique_obj_ids:
            self.object_filter.add_item(f'{obj_id}')

    def add_evaluation_results(self, eval_results: 'EvalResults') -> None:
        self._eval_results = eval_results
