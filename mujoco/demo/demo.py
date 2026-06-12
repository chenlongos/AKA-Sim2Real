"""Demo: sites + sensors on a simple pendulum."""
import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("demo.xml")
data = mujoco.MjData(model)

# Get site ID — used for fast lookup
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")

# Give it a push
data.qvel[0] = 5.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0
    while viewer.is_running():
        mujoco.mj_step(model, data)

        # --- Reading sensors ---
        # sensordata layout: [hinge_vel(1), tip_pos(3), tip_vel(3)]
        hinge_vel  = data.sensordata[0]      # joint velocity (rad/s)
        tip_pos    = data.sensordata[1:4]    # tip world xyz
        tip_vel    = data.sensordata[4:7]    # tip world velocity

        # site_xpos — same as framepos sensor (no noise)
        tip_world = data.site_xpos[site_id]

        # Print every 0.5s sim time
        if t % 250 == 0:
            print(f"t={data.time:.2f}s  "
                  f"angle={np.degrees(data.qpos[0]):+6.1f}°  "
                  f"vel={hinge_vel:+6.2f} rad/s  "
                  f"tip=({tip_world[0]:+.2f},{tip_world[1]:+.2f},{tip_world[2]:+.2f})")

        viewer.sync()
        t += 1
        time.sleep(1/500)
