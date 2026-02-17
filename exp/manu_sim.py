import numpy as np
import cv2
import robosuite as suite
from scipy.spatial.transform import Rotation as R

# 1. 環境作成
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview", 
    camera_widths=512,
    camera_heights=512,
    horizon=10000,
    control_freq=20,
)

env.sim.model.cam_pos[env.sim.model.camera_name2id("agentview")] = [0.8, 0.0, 1.0]

def get_state(obs):
    eef_pos = obs["robot0_eef_pos"]
    quat = obs["robot0_eef_quat"]
    r = R.from_quat(quat)
    euler = r.as_euler('xyz', degrees=False)
    gripper_state = np.mean(obs["robot0_gripper_qpos"], keepdims=True)
    return np.concatenate([eef_pos, euler, gripper_state]).astype(np.float32)

def main():
    obs = env.reset()
    keys = {'space': False}
    action_count = 0
    
    print("\n[グリップ維持・行動回数カウントモード]")
    print("操作中のみカウントが増えます。Spaceで掴んだまま移動可能です。")

    try:
        while True:
            # 常に最新の状態を反映させるためのアクション配列
            action = np.zeros(7)
            action[6] = 1.0 if keys['space'] else -1.0 
            
            has_input = False
            k = cv2.waitKey(1) & 0xFF
            
            # --- 入力判定 ---
            speed = 0.2
            if k == ord('w'): action[0] = -speed; has_input = True
            elif k == ord('s'): action[0] = speed; has_input = True
            elif k == ord('a'): action[1] = -speed; has_input = True
            elif k == ord('d'): action[1] = speed; has_input = True
            elif k == ord('r'): action[2] = speed; has_input = True
            elif k == ord('f'): action[2] = -speed; has_input = True
            elif k == ord(' '): 
                keys['space'] = not keys['space']
                action[6] = 1.0 if keys['space'] else -1.0
                has_input = True

            # 保存とリセット
            if k == 27: # ESC
                np.savez("franka_example_traj.npz", 
                         observations=np.expand_dims(np.flipud(obs["agentview_image"]), axis=0),
                         states=np.expand_dims(get_state(obs), axis=0))
                print(f"\n保存完了: {action_count} actions")
                break
            if k == ord('q'): 
                obs = env.reset(); keys['space'] = False; action_count = 0
                continue

            # --- 重要：シミュレーションは常に回す ---
            # これにより「何も押していない時」も物理演算（維持する力）が働く
            obs, reward, done, info = env.step(action)
            
            if has_input:
                action_count += 1

            # 表示
            img_rgb = np.flipud(obs["agentview_image"])
            frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            status = "CLOSED" if keys['space'] else "OPEN"
            cv2.putText(frame, f"Actions: {action_count} | Grip: {status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Panda Goal Capture", frame)

    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
