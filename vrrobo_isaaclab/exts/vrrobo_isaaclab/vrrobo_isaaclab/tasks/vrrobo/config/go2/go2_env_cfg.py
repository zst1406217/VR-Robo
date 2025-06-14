from __future__ import annotations

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
import vrrobo_isaaclab.tasks.vrrobo.mdp as mdp
import vrrobo_isaaclab.terrains as terrain_gen
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from vrrobo_isaaclab.assets import UNITREE_GO2_CFG  # isort: skip

##
# Scene definition
##

UPLEVEL_FREQUENCY = 5
BASE_HEIGHT = 0.32

GOAL = [
    (3.2 - 0.53, -0.43, -0.03 + 0.36 + BASE_HEIGHT),  # Red Cone
    (0, 0, 0),                                        # Green Cone
    (3.2 - 1.6, -1.59, -0.03 + 0.03 + BASE_HEIGHT),   # Blue Cone
]

ASSET_OFFSET = (3.2, 0.0, -0.01)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    object: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "object": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="./exts/scene_data/scene.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=ASSET_OFFSET, rot=(1.0, 0.0, 0.0, 0.0)),
            ),
            "wall_left": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Wall_1",
                spawn=sim_utils.CuboidCfg(
                    size=(3.4, 0.01, 1.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0 - 0.5, 0.0 + 1.9, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
            "wall_right": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Wall_2",
                spawn=sim_utils.CuboidCfg(
                    size=(3.4, 0.01, 1.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0 - 0.5, 0.0 - 3.5, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
            "wall_front": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Wall_3",
                spawn=sim_utils.CuboidCfg(
                    size=(0.01, 5.4, 1.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0 + 1.2, 0.0 - 0.8, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
            "wall_behind": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Wall_4",
                spawn=sim_utils.CuboidCfg(
                    size=(0.01, 5.4, 1.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0 - 2.2, 0.0 - 0.8, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        }
    )
    cone_red: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "cone_red": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cone_red",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="./exts/scene_data/red.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        }
    )
    cone_green: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "cone_green": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cone_green",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="./exts/scene_data/green.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        }
    )
    cone_blue: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "cone_blue": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cone_blue",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="./exts/scene_data/blue.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        }
    )


##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    rgb_command = mdp.RGBCommandCfg(
        resampling_time_range=(1e5, 1e5),
        RGB_prob=[1.0, 0.0, 0.0],
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.VelocityCommandActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        velocity_range=[0.5, 0.3, 1.0],
        policy_dir="../../../../../../low_level_policy/model.pt",
        uplevel_frequency=UPLEVEL_FREQUENCY,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        gs_image = ObsTerm(
            func=mdp.gs_image_feature,
            params={
                "camera_pos": [0.332 + 0.01, 0.0 + 0.015, 0.0578 + 0.095],
                "camera_rot": [0.0, 23.0, 0.0],
                "asset_offset_pos": ASSET_OFFSET,
                "asset_offset_rot": [1.0, 0.0, 0.0, 0.0],
            },
            noise=None,
        )
        goal_command = ObsTerm(func=mdp.rgb_command, params={"command_name": "rgb_command"}, noise=None)
        actions = ObsTerm(func=mdp.last_action, noise=None)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for Critic group."""

        actions = ObsTerm(func=mdp.last_action)
        root_pos_e = ObsTerm(func=mdp.head_pos_w)
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        goal_pos = ObsTerm(func=mdp.goal_pos_multi, params={"base_height": BASE_HEIGHT})
        goal_command = ObsTerm(func=mdp.rgb_command, params={"command_name": "rgb_command"}, noise=None)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=None)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=None)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=None)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=None)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=None, scale=0.05)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class LocomotionCfg(ObsGroup):
        """Observations for low-level locomotion."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel_zero, noise=None)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.zero_velocity_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})  # get raw action (joint position)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: ObsGroup = PolicyCfg()
    # critic observations groups
    critic: ObsGroup = CriticCfg()
    # locomotion observations groups
    locomotion: ObsGroup = LocomotionCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-1.5, 0.5), "yaw": (-1.0, 1.0)},
            "velocity_range": {
                "x": (0, 0),
                "y": (0, 0),
                "z": (0, 0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (0, 0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_asset = EventTerm(
        func=mdp.reset_asset,
        mode="reset",
    )

    reset_robot_with_cones = EventTerm(
        func=mdp.reset_cones,
        mode="reset",
        params={
            "pose_range": {
                "x": [(0.1, 3.0), (0.1, 3.0), (2.15, 3.0)],
                "y": [(0.2, 1.3), (-3.0, -1.9), (-1.05, -0.2)],
                "z": [(-0.0, -0.0), (-0.0, -0.0), (0.3456, 0.3456)],
                "yaw": [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)],
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    reach_goal = RewTerm(
        func=mdp.reach_goal,
        weight=0.5,
        params={"base_height": BASE_HEIGHT, "command_name": "rgb_command", "threshold": 0.35},
    )
    goal_dis = RewTerm(
        func=mdp.goal_dis, weight=5.0, params={"base_height": BASE_HEIGHT, "command_name": "rgb_command"}
    )
    goal_dis_z = RewTerm(
        func=mdp.goal_dis_z, weight=30.0, params={"base_height": BASE_HEIGHT, "command_name": "rgb_command"}
    )
    goal_heading = RewTerm(
        func=mdp.goal_heading_l1, weight=0.3, params={"base_height": BASE_HEIGHT, "command_name": "rgb_command"}
    )
    stand_still_at_goal = RewTerm(
        func=mdp.stand_still_at_goal, weight=1.0, params={"base_height": BASE_HEIGHT, "command_name": "rgb_command"}
    )
    track_lin_vel_xy_exp_command = RewTerm(
        func=mdp.track_lin_vel_xy_exp_command, weight=0.2, params={"std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp_command = RewTerm(
        func=mdp.track_ang_vel_z_exp_command, weight=0.2, params={"std": math.sqrt(0.25)}
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class UnitreeGo2GSBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=8)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 200 // UPLEVEL_FREQUENCY
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class UnitreeGo2GSEnvCfg(UnitreeGo2GSBaseEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.terrain.slope_threshold = 2
        self.scene.terrain.terrain_generator.horizontal_scale = 0.1
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.sub_terrains = {
            "perlin_terrain": terrain_gen.HfPerlinTerrainCfg(horizontal_scale=0.05, frequency=10, zScale=0.0),
        }

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)

        # no curriculum
        self.curriculum.terrain_levels = None


@configclass
class UnitreeGo2GSEnvCfg_PLAY(UnitreeGo2GSEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
