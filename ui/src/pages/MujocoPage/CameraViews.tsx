import {useEffect, useState, useRef, useCallback} from "react";
import {mujocoSocket, sendMujocoAction, sendMujocoCarAction, sendMujocoCameraMove, sendMujocoCameraZoom, onMujocoStateUpdate} from "../../api/socket";

interface MujocoState {
    car_pos?: number[];
    car_quat?: number[];
    arm_qpos?: number[];
    arm_qvel?: number[];
}

export const TopDownView = () => {
    const [image, setImage] = useState<string>("");
    const [dragging, setDragging] = useState(false);
    const lastPos = useRef<{x: number; y: number} | null>(null);

    useEffect(() => {
        const unsubscribe = onMujocoStateUpdate(mujocoSocket, (data) => {
            setImage(`data:image/jpeg;base64,${data.topdown}`);
        });
        return unsubscribe;
    }, []);

    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        setDragging(true);
        lastPos.current = {x: e.clientX, y: e.clientY};
    }, []);

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
        if (!dragging || !lastPos.current) return;
        const dx = e.clientX - lastPos.current.x;
        const dy = e.clientY - lastPos.current.y;
        lastPos.current = {x: e.clientX, y: e.clientY};
        sendMujocoCameraMove(mujocoSocket, dx, dy);
    }, [dragging]);

    const handleMouseUp = useCallback(() => {
        setDragging(false);
        lastPos.current = null;
    }, []);

    const handleWheel = useCallback((e: React.WheelEvent) => {
        sendMujocoCameraZoom(mujocoSocket, e.deltaY > 0 ? 1 : -1);
    }, []);

    return (
        <div className="bg-slate-900 rounded-lg overflow-hidden">
            <div className="px-3 py-2 border-b border-slate-700 flex items-center gap-2">
                <span className="w-2 h-2 bg-emerald-500 rounded-full"/>
                <span className="text-sm text-slate-300">俯视视角 (Top-Down)</span>
            </div>
            <div
                className="relative select-none"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onWheel={handleWheel}
                style={{cursor: dragging ? 'grabbing' : 'grab', userSelect: 'none'}}
            >
                {image ? (
                    <img src={image} alt="Top-Down View" className="w-full aspect-video object-cover pointer-events-none"/>
                ) : (
                    <div className="w-full aspect-video bg-slate-800 flex items-center justify-center">
                        <span className="text-slate-500">加载中...</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export const FirstPersonView = () => {
    const [image, setImage] = useState<string>("");

    useEffect(() => {
        const unsubscribe = onMujocoStateUpdate(mujocoSocket, (data) => {
            setImage(`data:image/jpeg;base64,${data.firstperson}`);
        });
        return unsubscribe;
    }, []);

    return (
        <div className="bg-slate-900 rounded-lg overflow-hidden">
            <div className="px-3 py-2 border-b border-slate-700 flex items-center gap-2">
                <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"/>
                <span className="text-sm text-slate-300">第一人称视角 (First-Person)</span>
            </div>
            <div className="relative">
                {image ? (
                    <img src={image} alt="First-Person View" className="w-full aspect-video object-cover"/>
                ) : (
                    <div className="w-full aspect-video bg-slate-800 flex items-center justify-center">
                        <span className="text-slate-500">加载中...</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export const ArmControl = () => {
    const [yaw, setYaw] = useState(0);
    const [pitch, setPitch] = useState(0);
    const [roll, setRoll] = useState(0);
    const [state, setState] = useState<MujocoState>({});

    useEffect(() => {
        const unsubscribe = onMujocoStateUpdate(mujocoSocket, (data) => {
            setState(data.state as MujocoState);
        });
        return unsubscribe;
    }, []);

    const handleArmAction = () => {
        sendMujocoAction(mujocoSocket, yaw, pitch, roll);
    };

    const armNames = ["Yaw", "Pitch", "Roll"];
    const armQpos = state.arm_qpos || [];
    const armQvel = state.arm_qvel || [];

    return (
        <div className="space-y-3">
            {/* 机械臂控制 */}
            <div className="bg-slate-900 rounded-lg p-4">
                <h3 className="text-sm font-medium text-slate-300 mb-3">机械臂控制</h3>
                <div className="space-y-3">
                    <div>
                        <label className="block text-xs text-slate-400 mb-1">Yaw (偏航): {yaw.toFixed(2)}</label>
                        <input
                            type="range"
                            min="-1"
                            max="1"
                            step="0.1"
                            value={yaw}
                            onChange={(e) => setYaw(parseFloat(e.target.value))}
                            className="w-full"
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-slate-400 mb-1">Pitch (俯仰): {pitch.toFixed(2)}</label>
                        <input
                            type="range"
                            min="-1"
                            max="1"
                            step="0.1"
                            value={pitch}
                            onChange={(e) => setPitch(parseFloat(e.target.value))}
                            className="w-full"
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-slate-400 mb-1">Roll (翻滚): {roll.toFixed(2)}</label>
                        <input
                            type="range"
                            min="-1"
                            max="1"
                            step="0.1"
                            value={roll}
                            onChange={(e) => setRoll(parseFloat(e.target.value))}
                            className="w-full"
                        />
                    </div>
                    <button
                        onClick={handleArmAction}
                        className="w-full px-3 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded text-sm transition-colors"
                    >
                        发送控制
                    </button>
                </div>
            </div>

            {/* 小车控制 */}
            <CarControl />

            {/* 关节状态显示 */}
            <div className="bg-slate-900 rounded-lg p-4">
                <h3 className="text-sm font-medium text-slate-300 mb-3">关节状态</h3>
                <div className="space-y-1 text-xs font-mono">
                    {armNames.map((name, i) => (
                        <div key={name} className="flex justify-between">
                            <span className="text-slate-400">{name}:</span>
                            <span className="text-slate-200">
                                pos={armQpos[i]?.toFixed(3) ?? "N/A"}, vel={armQvel[i]?.toFixed(3) ?? "N/A"}
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export const CarControl = () => {
    const [velLeft, setVelLeft] = useState(0);
    const [velRight, setVelRight] = useState(0);

    const handleCarAction = () => {
        sendMujocoCarAction(mujocoSocket, velLeft, velRight);
    };

    return (
        <div className="bg-slate-900 rounded-lg p-4">
            <h3 className="text-sm font-medium text-slate-300 mb-3">小车控制</h3>
            <div className="space-y-3">
                <div>
                    <label className="block text-xs text-slate-400 mb-1">左轮速度: {velLeft.toFixed(2)}</label>
                    <input
                        type="range"
                        min="-10"
                        max="10"
                        step="0.5"
                        value={velLeft}
                        onChange={(e) => setVelLeft(parseFloat(e.target.value))}
                        className="w-full"
                    />
                </div>
                <div>
                    <label className="block text-xs text-slate-400 mb-1">右轮速度: {velRight.toFixed(2)}</label>
                    <input
                        type="range"
                        min="-10"
                        max="10"
                        step="0.5"
                        value={velRight}
                        onChange={(e) => setVelRight(parseFloat(e.target.value))}
                        className="w-full"
                    />
                </div>
                <button
                    onClick={handleCarAction}
                    className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm transition-colors"
                >
                    发送控制
                </button>
            </div>
        </div>
    );
};