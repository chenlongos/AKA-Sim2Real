export const CAR_ARM_XML = `<mujoco>
    <option timestep="0.001" integrator="RK4"/>

    <default>
        <geom friction="1 0.1 0.1"/>
    </default>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 5" dir="0 0 -1" castshadow="true"/>

        <geom type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1" friction="1 0.1 0.1"/>

        <body name="topdown_cam" pos="0 0 8">
            <camera name="topdown" euler="0 0 0" fovy="45"/>
            <geom type="sphere" size="0.12" rgba="1 0.2 0.2 0.7" contype="0" conaffinity="0"/>
        </body>

        <body name="car" pos="0 0 0.22">
            <camera name="firstperson" pos="0.9 0 0.2" euler="0 0 0" fovy="110"/>
            <geom type="sphere" pos="0.4 0 0.2" size="0.05" rgba="0.2 1 0.2 0.7" contype="0" conaffinity="0"/>

            <geom type="box" size="0.5 0.3 0.15" rgba="0.2 0.6 0.8 1" mass="10"/>

            <body name="wheel_fl" pos="0.4 0.3 -0.1">
                <joint name="wheel_fl_joint" type="hinge" axis="0 1 0" range="-100 100"/>
                <geom type="cylinder" size="0.12 0.05" fromto="0 -0.05 0 0 0.05 0" rgba="0.1 0.1 0.1 1" mass="1"/>
            </body>
            <body name="wheel_fr" pos="0.4 -0.3 -0.1">
                <joint name="wheel_fr_joint" type="hinge" axis="0 1 0" range="-100 100"/>
                <geom type="cylinder" size="0.12 0.05" fromto="0 -0.05 0 0 0.05 0" rgba="0.1 0.1 0.1 1" mass="1"/>
            </body>
            <body name="wheel_rl" pos="-0.4 0.3 -0.1">
                <joint name="wheel_rl_joint" type="hinge" axis="0 1 0" range="-100 100"/>
                <geom type="cylinder" size="0.12 0.05" fromto="0 -0.05 0 0 0.05 0" rgba="0.1 0.1 0.1 1" mass="1"/>
            </body>
            <body name="wheel_rr" pos="-0.4 -0.3 -0.1">
                <joint name="wheel_rr_joint" type="hinge" axis="0 1 0" range="-100 100"/>
                <geom type="cylinder" size="0.12 0.05" fromto="0 -0.05 0 0 0.05 0" rgba="0.1 0.1 0.1 1" mass="1"/>
            </body>

            <body name="arm_base" pos="-0.2 0 0.15">
                <joint name="arm_yaw" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom type="box" size="0.1 0.1 0.15" rgba="0.7 0.3 0.3 1" mass="2"/>

                <body name="arm_link1" pos="0 0 0.2">
                    <joint name="arm_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                    <geom type="box" size="0.06 0.06 0.3" rgba="0.3 0.7 0.3 1" mass="1"/>

                    <body name="arm_link2" pos="0 0 0.35">
                        <joint name="arm_roll" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
                        <geom type="box" size="0.05 0.05 0.25" rgba="0.3 0.3 0.7 1" mass="0.8"/>

                        <body name="arm_wrist" pos="0 0 0.3">
                            <joint name="arm_wrist" type="ball"/>
                            <geom type="box" size="0.04 0.04 0.15" rgba="0.9 0.9 0.2 1" mass="0.3"/>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor joint="arm_yaw" name="motor_yaw" gear="50" ctrllimited="true" ctrlrange="-10 10"/>
        <motor joint="arm_pitch" name="motor_pitch" gear="50" ctrllimited="true" ctrlrange="-10 10"/>
        <motor joint="arm_roll" name="motor_roll" gear="30" ctrllimited="true" ctrlrange="-10 10"/>
        <motor joint="wheel_fl_joint" name="motor_wheel_fl" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
        <motor joint="wheel_fr_joint" name="motor_wheel_fr" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
        <motor joint="wheel_rl_joint" name="motor_wheel_rl" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
        <motor joint="wheel_rr_joint" name="motor_wheel_rr" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
    </actuator>
</mujoco>`;
