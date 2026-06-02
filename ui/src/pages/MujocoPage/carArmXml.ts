export const CAR_ARM_XML = `<mujoco>
    <option timestep="0.001" integrator="RK4"/>

    <default>
        <geom friction="1 0.1 0.1"/>
    </default>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 5" dir="0 0 -1" castshadow="true"/>

        <geom type="plane" size="100 100 0.1" rgba="0.8 0.8 0.8 1" friction="1.5 0.005 0.0001"/>

        <!-- Maze: corridors 2m wide, wall height 0.6m, thickness 0.1m -->
        <!-- Outer boundary -->
        <geom type="box" pos=" 0   -4 0.3" size=" 4   0.05 0.3" rgba="0.5 0.3 0.1 1"/>
        <geom type="box" pos=" 0    4 0.3" size=" 4   0.05 0.3" rgba="0.5 0.3 0.1 1"/>
        <geom type="box" pos="-4    0 0.3" size="0.05  4   0.3" rgba="0.5 0.3 0.1 1"/>
        <geom type="box" pos=" 4    0 0.3" size="0.05  4   0.3" rgba="0.5 0.3 0.1 1"/>

        <!-- Inner walls -->
        <!-- V: x=-2, y=-3..1  (gap at y=1..3) -->
        <geom type="box" pos="-2  -1   0.3" size="0.05 2   0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- V: x=-2, y=1..3 -->
        <geom type="box" pos="-2   2   0.3" size="0.05 1   0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- V: x=0,  y=-2..1  (gap at y=1..3, y=-4..-2) -->
        <geom type="box" pos=" 0  -0.5 0.3" size="0.05 1.5 0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- V: x=0,  y=1..3 -->
        <geom type="box" pos=" 0   2   0.3" size="0.05 1   0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- V: x=2,  y=-3..-1 -->
        <geom type="box" pos=" 2  -2   0.3" size="0.05 1   0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- V: x=2,  y=0..3 -->
        <geom type="box" pos=" 2   1.5 0.3" size="0.05 1.5 0.3" rgba="0.6 0.4 0.2 1"/>

        <!-- H: y=-1, x=-2..1 (gap at x=1..2 for car to pass) -->
        <geom type="box" pos="-0.5 -1 0.3" size=" 1.5 0.05 0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- H: y= 1, x=-4..-1 -->
        <geom type="box" pos="-2.5 1   0.3" size=" 1.5 0.05 0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- H: y= 1, x= 1..4 -->
        <geom type="box" pos=" 2.5 1   0.3" size=" 1.5 0.05 0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- H: y= 3, x= 0..4 -->
        <geom type="box" pos=" 2   3   0.3" size=" 2   0.05 0.3" rgba="0.6 0.4 0.2 1"/>

        <body name="car" pos="3 -3 0.22">
            <freejoint/>

            <body name="camera_body" pos="0.4 0 0.2" euler="0 0 -90">
                <camera name="firstperson" pos="0 0 0" euler="0 0 0" fovy="110"/>
                <geom type="box" pos="0 0 0" size="0.05 0.1 0.05" rgba="0 0 0 1" contype="0" conaffinity="0"/>
                <geom type="cylinder" fromto="0 0.15 0 0 0 0" size="0.02" rgba="0 1 0 1" contype="0" conaffinity="0"/>
            </body>

            <geom type="box" size="0.5 0.3 0.15" rgba="0.2 0.6 0.8 1" mass="10"/>

            <body name="wheel_fl" pos="0.4 0.3 -0.1">
                <joint name="wheel_fl_joint" type="hinge" axis="0 1 0"/>
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
        <motor joint="wheel_fl_joint" name="motor_wheel_fl" gear="2" ctrllimited="true" ctrlrange="-5 5"/>
        <motor joint="wheel_fr_joint" name="motor_wheel_fr" gear="2" ctrllimited="true" ctrlrange="-5 5"/>
        <motor joint="wheel_rl_joint" name="motor_wheel_rl" gear="2" ctrllimited="true" ctrlrange="-5 5"/>
        <motor joint="wheel_rr_joint" name="motor_wheel_rr" gear="2" ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>
</mujoco>`;