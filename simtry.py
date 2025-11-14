import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("Spider_v1.xml")
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "start_pose")
mujoco.mj_resetDataKeyframe(model, data, key_id)

target_qpos = np.array([
    0.0,  0.64411, -1.86921,
    0.0,  0.64411, -1.86921,
    0.0,  0.64411, -1.86921,
    0.0,  0.64411, -1.86921
])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = target_qpos
        print(data.qpos)
        mujoco.mj_step(model, data)
        viewer.sync()
