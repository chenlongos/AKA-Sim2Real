export const MAZE_XML = `
        <geom type="plane" size="100 100 0.1" rgba="0.8 0.8 0.8 1" friction="1.5 0.005 0.0001"/>

        <!-- Maze: corridors 2m wide, wall height 0.6m, thickness 0.1m -->
        <!-- Outer boundary -->
        <geom type="box" pos=" 0   -4 0.3" size=" 4   0.05 0.3" rgba="0.5 0.3 0.1 1"/>
        <geom type="box" pos=" 0    4 0.3" size=" 4   0.05 0.3" rgba="0.5 0.3 0.1 1"/>
        <geom type="box" pos="-4    0 0.3" size="0.05  4   0.3" rgba="0.5 0.3 0.1 1"/>
        <geom type="box" pos=" 4    0 0.3" size="0.05  4   0.3" rgba="0.5 0.3 0.1 1"/>

        <!-- Inner walls -->
        <!-- V: x=-2, y=-3..1 -->
        <geom type="box" pos="-2  -1   0.3" size="0.05 2   0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- V: x=-2, y=1..3 -->
        <geom type="box" pos="-2   2   0.3" size="0.05 1   0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- V: x=0,  y=-2..1 -->
        <geom type="box" pos=" 0  -0.5 0.3" size="0.05 1.5 0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- V: x=0,  y=1..3 -->
        <geom type="box" pos=" 0   2   0.3" size="0.05 1   0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- V: x=2,  y=-3..-1 -->
        <geom type="box" pos=" 2  -2   0.3" size="0.05 1   0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- V: x=2,  y=0..3 -->
        <geom type="box" pos=" 2   1.5 0.3" size="0.05 1.5 0.3" rgba="0.6 0.4 0.2 1"/>

        <!-- H: y=-1, x=-2..1 -->
        <geom type="box" pos="-0.5 -1 0.3" size=" 1.5 0.05 0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- H: y= 1, x=-4..-1 -->
        <geom type="box" pos="-2.5 1   0.3" size=" 1.5 0.05 0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- H: y= 1, x= 1..4 -->
        <geom type="box" pos=" 2.5 1   0.3" size=" 1.5 0.05 0.3" rgba="0.6 0.4 0.2 1"/>
        <!-- H: y= 3, x= 0..4 -->
        <geom type="box" pos=" 2   3   0.3" size=" 2   0.05 0.3" rgba="0.6 0.4 0.2 1"/>
`;