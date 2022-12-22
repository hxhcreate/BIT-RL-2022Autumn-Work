import csv
import matplotlib.pyplot as plt        

def draw():
    csv_reader1 = csv.reader(open("./origin/record.csv"))
    episode = []
    duration = []
    reward_origin = []
    for line in csv_reader1:
        episode.append(int(line[0]))
        duration.append(int(line[1]))
        reward_origin.append(float(line[2]))

    csv_reader2 = csv.reader(open("./C=10/record.csv"))
    episode_T = []
    duration_T = []
    reward_T = []
    for line in csv_reader2:
        episode_T.append(int(line[0]))
        duration_T.append(int(line[1]))
        reward_T.append(float(line[2]))
    plt.figure(dpi=500)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    ax = plt.subplot(111)
    ax.set_title('DQN')
    ax.plot(reward_origin, label="origin")
    ax.plot(reward_T, label="target")    
    ax.legend()
    ax.set_title("C=10")              
    plt.savefig("./C=10/result.jpg")
    plt.show()


if __name__ == "__main__":
    draw()