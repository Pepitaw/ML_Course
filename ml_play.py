import pickle
import numpy as np
from mlgame.communication import ml as comm
import os.path as path

# cd C:\Users\user\Desktop\課程\大二下\基於遊戲的機器學習\MLGame-master
# python MLGame.py -i ml_play.py pingpong HARD 10


# 'frame': 10, 'status': 'GAME_ALIVE', 'ball': (35, 143), 'ball_speed': (-7, 7), 'platform_1P': (35,420), 'platform_2P': (35, 50),
# 'blocker': (110, 240), 'command_1P': 'MOVE_LEFT', 'command_2P': 'MOVE_LEFT'}
# Data = [Commands, Balls, Ball_speed, PlatformPos, Blocker, vectors, direction]

#...............Start the game...............#


def ml_loop(side: str):

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.
    ball_served = False
    filename = path.join(path.dirname(__file__),
                         "random_forest.pickle")
    with open(filename, 'rb') as file:
        clf = pickle.load(file)
    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()

    s = [93, 93]

    def get_direction(ball_x, ball_y, ball_pre_x, ball_pre_y):
        VectorX = ball_x - ball_pre_x
        VectorY = ball_y - ball_pre_y
        if(VectorX > 0 and VectorY > 0):
            return 0
        elif(VectorX > 0 and VectorY < 0):
            return 1
        elif(VectorX < 0 and VectorY > 0):
            return 2
        elif(VectorX < 0 and VectorY < 0):
            return 3
        else:
            return 4

    # 3. Start an endless loop.
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.recv_from_game()
        #Data = [Commands, Balls, Ball_speed, PlatformPos, Blocker, vectors, direction]
        #Feature = [Balls, Ball_speed, PlatformPos, Blocker, direction]
        feature = []
        for i in range(0, 2):
            # feature.append(scene_info["ball"][i])
            # feature.append(scene_info["ball_speed"][i])
            feature.append(scene_info["platform_1P"][i])
            feature.append(scene_info["blocker"][i])
        feature.append(feature[0] - s[0])
        feature.append(feature[1] - s[1])
        feature.append(get_direction(feature[0], feature[1], s[0], s[1]))
        s = [feature[0], feature[1]]
        # print(feature)
        feature = np.array(feature)
        feature = feature.reshape((-2, 9))

        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info["status"] != "GAME_ALIVE":
            # Do some stuff if needed
            ball_served = False

            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()
            continue

        # 3.3. Put the code here to handle the scene information

        # 3.4. Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_to_game(
                {"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})
            ball_served = True
        else:
            y = clf.predict(feature)
            if scene_info["platform_1P"][0]+20 > (pred-10) and scene_info["platform_1P"][0]+20 < (pred+10):
                move = 0  # NONE
            elif scene_info["platform_1P"][0]+20 <= (pred-10):
                move = 1  # goes right
            else:
                move = 2  # goes left
            if move == 0:
                comm.send_to_game(
                    {"frame": scene_info["frame"], "command": "NONE"})
            elif move == 1:
                comm.send_to_game(
                    {"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
            else:
                comm.send_to_game(
                    {"frame": scene_info["frame"], "command": "MOVE_LEFT"})
