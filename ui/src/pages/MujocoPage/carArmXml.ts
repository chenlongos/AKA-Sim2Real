import { MAZE_XML } from './mazeXml';

export const CAR_ARM_XML = `<mujoco>
    <option timestep="0.001" integrator="RK4" gravity="0 0 -15"/>

    <default>
        <geom friction="1 0.1 0.1"/>
    </default>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 5" dir="0 0 -1" castshadow="true"/>
        ${MAZE_XML}

        <body name="car" pos="3 -3 0.22">
            <freejoint/>

            <body name="camera_body" pos="0.4 0 0.2" euler="0 0 -90">
                <camera name="firstperson" pos="0 0 0" euler="0 0 0" fovy="110"/>
                <geom type="box" pos="0 0 0" size="0.05 0.1 0.05" rgba="0 0 0 1" contype="0" conaffinity="0"/>
                <geom type="cylinder" fromto="0 0.15 0 0 0 0" size="0.02" rgba="0 1 0 1" contype="0" conaffinity="0"/>
            </body>

            <geom type="box" size="0.5 0.3 0.15" rgba="0.2 0.6 0.8 1" mass="10"/>

            <body name="wheel_fl" pos="0.4 0.3 -0.1">
                <joint name="wheel_fl_joint" type="hinge" axis="0 1 0" damping="0.1"/>
                <geom type="cylinder" size="0.12 0.05" euler="90 0 0" rgba="0.1 0.1 0.1 1" mass="1" friction="1.5 0.005 0.0001"/>
            </body>
            <body name="wheel_fr" pos="0.4 -0.3 -0.1">
                <joint name="wheel_fr_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.12 0.05" euler="90 0 0" rgba="0.1 0.1 0.1 1" mass="1" friction="1.5 0.005 0.0001"/>
            </body>
            <body name="wheel_rl" pos="-0.4 0.3 -0.1">
                <joint name="wheel_rl_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.12 0.05" euler="90 0 0" rgba="0.1 0.1 0.1 1" mass="1" friction="1.5 0.005 0.0001"/>
            </body>
            <body name="wheel_rr" pos="-0.4 -0.3 -0.1">
                <joint name="wheel_rr_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.12 0.05" euler="90 0 0" rgba="0.1 0.1 0.1 1" mass="1" friction="1.5 0.005 0.0001"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor joint="wheel_fl_joint" name="motor_wheel_fl" gear="3" ctrllimited="true" ctrlrange="-5 5"/>
        <motor joint="wheel_fr_joint" name="motor_wheel_fr" gear="3" ctrllimited="true" ctrlrange="-5 5"/>
        <motor joint="wheel_rl_joint" name="motor_wheel_rl" gear="3" ctrllimited="true" ctrlrange="-5 5"/>
        <motor joint="wheel_rr_joint" name="motor_wheel_rr" gear="3" ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>
</mujoco>`;