import {useEffect, useState} from "react";
import {mujocoSocket} from "../../api/socket";
import {TopDownView, FirstPersonView} from "./CameraViews";

export default function MujocoPage() {
    const [connected, setConnected] = useState(false);

    useEffect(() => {
        mujocoSocket.on("connect", () => {
            setConnected(true);
        });
        mujocoSocket.on("disconnect", () => {
            setConnected(false);
        });

        return () => {
            mujocoSocket.off("connect");
            mujocoSocket.off("disconnect");
        };
    }, []);

    return (
        <div className="flex flex-col h-screen bg-slate-950 overflow-hidden">
            {/* 顶部标题栏 */}
            <div className="flex items-center justify-between px-6 py-2 bg-slate-900/50 border-b border-slate-800">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-linear-to-br from-violet-600 to-blue-600 flex items-center justify-center shadow-lg shadow-violet-900/20">
                        <span className="text-white font-bold text-sm">MJC</span>
                    </div>
                    <div>
                        <h2 className="text-base font-bold text-slate-100">MuJoCo 小车+机械臂</h2>
                        <p className="text-xs text-slate-500">3D Physics Simulation</p>
                    </div>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/50 border border-slate-700">
                        <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-500' : 'bg-red-500'}`}/>
                        <span className="text-xs text-slate-300">{connected ? 'Connected' : 'Disconnected'}</span>
                    </div>
                </div>
            </div>

            {/* 主内容区 */}
            <div className="flex flex-1 overflow-hidden p-4 gap-4">
                {/* 中间 - 俯视视角 */}
                <div className="flex-1 min-w-0">
                    <TopDownView />
                </div>

                {/* 右侧 - 第一人称视角 */}
                <div className="w-96 flex flex-col min-w-0">
                    <FirstPersonView />
                </div>
            </div>
        </div>
    );
}