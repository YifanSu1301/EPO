import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

task = 'reorientation'

# Load the SAPG data
SAPG_6_frames = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/sapg_6/frames.csv")
SAPG_6_success = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/sapg_6/success.csv")
SAPG_64_frames = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/sapg_64/frames.csv")
SAPG_64_success = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/sapg_64/success.csv")

# Load the ppo data
ppo_frames = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/ppo/frames.csv")
ppo_success = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/ppo/success.csv")

# Load Evolution data
Evolution_8_frames = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/evolution_8/frames.csv")
Evolution_8_success = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/evolution_8/success.csv")
Evolution_64_frames = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/evolution_64/frames.csv")
Evolution_64_success = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/evolution_64/success.csv")
Evolutin_16_frames = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/evolution_16/frames.csv")
Evolution_16_success = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/evolution_16/success.csv")
Evolution_32_frames = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/evolution_32/frames.csv")
Evolution_32_success = pd.read_csv(f"/home/yifan/Projects/sapg/result/{task}/evolution_32/success.csv")


def draw_plot(plot, frames, success, name):
    # Interpolate the frames
    steps = frames['Step']
    frames = frames.iloc[:, 1]

    # add 0 to the front of frames
    combined_df = pd.DataFrame({'step': steps, 'frame': frames})
    complete_steps = pd.DataFrame({'step': range(0, combined_df['step'].max() + 1)})
    merged_df = pd.merge(complete_steps, combined_df, on='step', how='left')
    frames_interpolate = merged_df.interpolate()

    frame_list = []
    mean_success_list = []
    low_std_success_list = []
    high_std_success_list = []
    success_num = len(success.iloc[:, 1])
    
    for i in range(success_num):
        # get the 0,3,6,9,12 of success
        
        num_run = len(success.iloc[0])//3
        successes = success.iloc[i, [j*3+1 for j in range(num_run)]]
        curr_step = success.iloc[i, 0]
        curr_frame = frames_interpolate[frames_interpolate['step'] == curr_step]
        if curr_frame.empty:
            continue
        # if curr_frame.iloc[0, 1] is none
        if curr_frame.iloc[0, 1] > 5e9:
            break
        # append values
        frame_list.append(curr_frame.iloc[0, 1])
        mean_success_list.append(successes.mean())
        low_std_success_list.append(successes.mean() - successes.std())
        high_std_success_list.append(successes.mean() + successes.std())
    
    # smooth the mean_success_list
    mean_success_list = np.convolve(mean_success_list, np.ones(10)/10, mode='valid')
    low_std_success_list = np.convolve(low_std_success_list, np.ones(10)/10, mode='valid')
    high_std_success_list = np.convolve(high_std_success_list, np.ones(10)/10, mode='valid')
    frame_list = frame_list[:len(mean_success_list)]

    plot.plot(frame_list, mean_success_list, label=name)
    plot.fill_between(frame_list, low_std_success_list, high_std_success_list, alpha=0.2)
    # set legend
    return 

if __name__ == "__main__":
    # draw the graph
    fig, ax = plt.subplots()


    draw_plot(ax, ppo_frames, ppo_success, 'PPO')
    draw_plot(ax, SAPG_6_frames, SAPG_6_success, 'SAPG_6')
    draw_plot(ax, SAPG_64_frames, SAPG_64_success, 'SAPG_64')
    # draw_plot(ax, Evolution_8_frames, Evolution_8_success, 'Evolution_8')
    # draw_plot(ax, Evolutin_16_frames, Evolution_16_success, 'Evolution_16')
    # draw_plot(ax, Evolution_32_frames, Evolution_32_success, 'Evolution_32')
    draw_plot(ax, Evolution_64_frames, Evolution_64_success, 'Evolution_64')

    ax.set(xlabel='Number of envsteps', ylabel='Episode successes',)
    # ignore 1,3,5,7,9 legend
    ax.legend()
    ax.grid()
    # set the title
    ax.set_title(f"{task}")
    plt.savefig(f"/home/yifan/Projects/sapg/result/{task}/graph(comparison).png")


# # Interpolate the frames
# SAPG_steps = SAPG_frames['Step']
# SAPG_frames = SAPG_frames.iloc[:, 1]
# # add 0 to the front of SAPG_frames
# combined_df = pd.DataFrame({'step': SAPG_steps, 'frame': SAPG_frames})
# complete_steps = complete_steps = pd.DataFrame({'step': range(0, combined_df['step'].max() + 1)})
# merged_df = pd.merge(complete_steps, combined_df, on='step', how='left')
# SAPG_frames_interpolate = merged_df.interpolate()


# # Get the corresponding success
# SAPG_frame_list = []
# SAPF_success_list = []
# SAPG_suceess_num = len(SAPG_success.iloc[:, 1])

# for i in range(SAPG_suceess_num):
#     success = SAPG_success.iloc[i, 1]
#     curr_step = SAPG_success.iloc[i, 0]
#     curr_frame = SAPG_frames_interpolate[SAPG_frames_interpolate['step'] == curr_step]
#     if curr_frame.empty:
#         continue
#     if curr_frame.iloc[0, 1] > 2e10:
#         break
#     # append values
#     SAPG_frame_list.append(curr_frame.iloc[0, 1])
#     SAPF_success_list.append(SAPG_success.iloc[i, 1])

# # Load the SAPG data
# SAPG_frames = pd.read_csv(f"/home/yifan/Projects/sapg/IsaacGymEnvs/result/{task}/SAPG_60/frames.csv")
# SAPG_success = pd.read_csv(f"/home/yifan/Projects/sapg/IsaacGymEnvs/result/{task}/SAPG_60/success.csv")

# # Interpolate the frames
# SAPG_steps = SAPG_frames['Step']
# SAPG_frames = SAPG_frames.iloc[:, 1]
# # add 0 to the front of SAPG_frames
# combined_df = pd.DataFrame({'step': SAPG_steps, 'frame': SAPG_frames})
# complete_steps = complete_steps = pd.DataFrame({'step': range(0, combined_df['step'].max() + 1)})
# merged_df = pd.merge(complete_steps, combined_df, on='step', how='left')
# SAPG_frames_interpolate = merged_df.interpolate()


# # Get the corresponding success
# SAPG60_frame_list = []
# SAPG60_success_list = []
# SAPG60_suceess_num = len(SAPG_success.iloc[:, 1])
# print(SAPG60_suceess_num)

# for i in range(SAPG60_suceess_num):
#     # print(i)
#     success = SAPG_success.iloc[i, 1]
#     curr_step = SAPG_success.iloc[i, 0]
#     curr_frame = SAPG_frames_interpolate[SAPG_frames_interpolate['step'] == curr_step]
#     if curr_frame.empty:
#         continue
#     if curr_frame.iloc[0, 1] > 2e10:
#         break
#     # append values
#     SAPG60_frame_list.append(curr_frame.iloc[0, 1])
#     SAPG60_success_list.append(SAPG_success.iloc[i, 1])


# # Load the SAPG data
# SAPG_frames = pd.read_csv(f"/home/yifan/Projects/sapg/IsaacGymEnvs/result/{task}/Evolution_60/frames.csv")
# SAPG_success = pd.read_csv(f"/home/yifan/Projects/sapg/IsaacGymEnvs/result/{task}/Evolution_60/success.csv")

# # Interpolate the frames
# SAPG_steps = SAPG_frames['Step']
# SAPG_frames = SAPG_frames.iloc[:, 1]
# # add 0 to the front of SAPG_frames
# combined_df = pd.DataFrame({'step': SAPG_steps, 'frame': SAPG_frames})
# complete_steps = complete_steps = pd.DataFrame({'step': range(0, combined_df['step'].max() + 1)})
# merged_df = pd.merge(complete_steps, combined_df, on='step', how='left')
# SAPG_frames_interpolate = merged_df.interpolate()


# # Get the corresponding success
# Evolution_frame_list = []
# Evolution_success_list = []
# Evolution_suceess_num = len(SAPG_success.iloc[:, 1])

# for i in range(Evolution_suceess_num):
#     success = SAPG_success.iloc[i, 1]
#     curr_step = SAPG_success.iloc[i, 0]
#     curr_frame = SAPG_frames_interpolate[SAPG_frames_interpolate['step'] == curr_step]
#     if curr_frame.empty:
#         continue
#     if curr_frame.iloc[0, 1] > 2e10:
#         break
#     # append values
#     Evolution_frame_list.append(curr_frame.iloc[0, 1])
#     Evolution_success_list.append(SAPG_success.iloc[i, 1])
# # draw the graph
# import matplotlib.pyplot as plt
# import numpy as np

# fig, ax = plt.subplots()

# ax.plot(SAPG_frame_list, SAPF_success_list)
# ax.plot(SAPG60_frame_list, SAPG60_success_list)
# ax.plot(Evolution_frame_list, Evolution_success_list)

# ax.set(xlabel='Frames', ylabel='Success Rate',
#     title=f"{task}")

# ax.legend(['SAPG_6', 'SAPG_60', 'Evolution_60'])

# ax.grid()
# plt.show()

    # draw the graph
# import matplotlib.pyplot as plt
# import numpy as np
    
# fig, ax = plt.subplots()

# # Load the Evolution data
# Evolution_frame_list_total = []
# Evolution_success_list_total = []

# for i in range(8):
#     Evolution_frames = pd.read_csv(f'/home/yifan/Projects/sapg/IsaacGymEnvs/result/{task}/Evolution/0{i}/frames.csv')
#     Evolution_success = pd.read_csv(f'/home/yifan/Projects/sapg/IsaacGymEnvs/result/{task}/Evolution/0{i}/success.csv')

#     # Interpolate the frames
#     Evolution_steps = Evolution_frames['Step']
#     Evolution_frames = Evolution_frames.iloc[:, 1]
#     # add 0 to the front of Evolution_frames
#     combined_df = pd.DataFrame({'step': Evolution_steps, 'frame': Evolution_frames})
#     complete_steps = complete_steps = pd.DataFrame({'step': range(0, combined_df['step'].max() + 1)})
#     merged_df = pd.merge(complete_steps, combined_df, on='step', how='left')
#     Evolution_frames_interpolate = merged_df.interpolate()

#     # Get the corresponding success
#     Evolution_frame_list = []
#     Evolution_success_list = []
#     Evolution_suceess_num = len(Evolution_success.iloc[:, 1])

#     for i in range(Evolution_suceess_num):
#         success = Evolution_success.iloc[i, 1]
#         curr_step = Evolution_success.iloc[i, 0]
#         curr_frame = Evolution_frames_interpolate[Evolution_frames_interpolate['step'] == curr_step]
#         if curr_frame.empty:
#             continue
#         if curr_frame.iloc[0, 1] > 5e9:
#             break
#         # append values
#         Evolution_frame_list.append(curr_frame.iloc[0, 1]*8)
#         Evolution_success_list.append(Evolution_success.iloc[i, 1])
    



#     ax.plot(Evolution_frame_list, Evolution_success_list)


# ax.plot(SAPG_frame_list, SAPF_success_list)
# ax.set(xlabel='Frames', ylabel='Success Rate',
#     title=f"{task}")

# # ax.legend(['Evolution', 'SAPG'])

# ax.grid()
# plt.show()

    

