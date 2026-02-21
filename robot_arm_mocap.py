"""
Panda robot control using MuJoCo weld equality constraint.
The weld constraint rigidly attaches the hand body to the mocap body,
allowing mouse-driven control via the passive viewer.
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("panda_pick_and_place.xml")
data = mujoco.MjData(model)

# IDs
mocap_id  = model.body("mocap").mocapid[0]
weld_id   = model.equality("weld_hand").id

with mujoco.viewer.launch_passive(model=model, data=data) as viewer:

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    data.eq_active[weld_id] = 1
    
    hand_id = model.body("hand").id
    data.mocap_pos[mocap_id]  = data.xpos[hand_id].copy()
    data.mocap_quat[mocap_id] = data.xquat[hand_id].copy()

    while viewer.is_running():
        step_start = time.time()

        data.ctrl[:7] = data.qfrc_bias[:7]

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)