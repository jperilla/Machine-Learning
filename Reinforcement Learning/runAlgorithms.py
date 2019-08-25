import re
import sys

from raceTrack import RaceTrack


def run_race_track_simulator_value_iteration(restart_on_crash, track_file_path, run_value_iteration=False):
    """
    This function runs the race track simulator
    :param restart_on_crash: True if you should continue from nearest current location on crash, False restart at
    starting position on crash
    :param track_file_path: The file path to the race track text file
    :param run_value_iteration: True if you want to run the value iteration to create the policy, if False
    it will attempt to read the policy from a file
    """
    print(f"Running Race Track simulator with options: Restart on crash? {'yes' if restart_on_crash else 'no'}, "
          f"Track file: {track_file_path}")

    print("Race Track")
    with open(track_file_path, "r") as file:
        max_x, max_y = file.readline().split(',')
        race_track = file.read().splitlines()
        print(race_track)
        starting_positions = find_starting_positions(race_track)
        track_simulator = RaceTrack(race_track, track_file_path[track_file_path.rindex('/')+1:],
                                    starting_positions[1][0], starting_positions[1][1], max_x, max_y)
        if run_value_iteration:
            track_simulator.learn_value_iteration()
        else:
            track_simulator.run_racetrack_on_value_iteration(restart_on_crash)


def run_race_track_simulator_sarsa(restart_on_crash, track_file_path, run_value_iteration=False):
    """
        This function runs the race track simulator
        :param restart_on_crash: True if you should continue from nearest current location on crash, False restart at
        starting position on crash
        :param track_file_path: The file path to the race track text file
        :param run_value_iteration: True if you want to run the value iteration to create the policy, if False
        it will attempt to read the policy from a file
        """
    print(f"Running Race Track simulator with options: Restart on crash? {'yes' if restart_on_crash else 'no'}, "
          f"Track file: {track_file_path}")

    print("Race Track")
    with open(track_file_path, "r") as file:
        max_x, max_y = file.readline().split(',')
        race_track = file.read().splitlines()
        print(race_track)
        starting_positions = find_starting_positions(race_track)
        track_simulator = RaceTrack(race_track, track_file_path[track_file_path.rindex('/') + 1:],
                                    starting_positions[1][0], starting_positions[1][1], max_x, max_y)

        track_simulator.run_racetrack_on_sarsa(3, restart_on_crash)


def find_starting_positions(track):
    """
        This function takes a track and finds the instances of S and returns these start positions with a list of tuples
        of (x, y) pair values
        :param track: A list of strings that make up a grid that has a set of S values somewhere that indicate the
        starting line
        :return: A list of tuples of (x,y) pair values that represent possible starting positions
        """
    return find_positions(track, 'S')


def find_finish_positions(track):
    """
        This function takes a track and finds the instances of F and returns these finish positions with a list of tuples
        of (x, y) pair values
        :param track: A list of strings that make up a grid that has a set of S values somewhere that indicate the
        finish line
        :return: A list of tuples of (x,y) pair values that represent possible finish positions
        """
    return find_positions(track, 'S')


def find_positions(list_of_sentences, word):
    """
    This function finds all occurrences of a word in a list of sentences
    :param list_of_sentences: The list of sentences to find occurrences of a word in
    :param word: The word or character to find
    :return: A list of tuples with all positions
    """
    start = list()
    for index, row in enumerate(list_of_sentences):
        for match in re.finditer('S', row):
            start.append(tuple((index, match.start())))

    return start


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Please add options (--crash-continue --crash-restart) and (--value or --sarsa) "
              "and (--create-policy --run)"
              "and the full file path to the race track file")
        print("Example: python runAlgorithms.py --crash-continue --value --run ./data/L-track.txt ")
        exit()

    if sys.argv[1] == "--crash-continue":
        if sys.argv[2] == "--value" and sys.argv[3] == "--create-policy":
            run_race_track_simulator_value_iteration(False, sys.argv[4], True)
            exit()
        elif sys.argv[2] == "--value" and sys.argv[3] == "--run":
            run_race_track_simulator_value_iteration(False, sys.argv[4], False)
            exit()
        elif sys.argv[2] == "--sarsa":
            run_race_track_simulator_sarsa(False, sys.argv[4])
            exit()

    if sys.argv[1] == "--crash-restart":
        if sys.argv[2] == "--value" and sys.argv[3] == "--create-policy":
            run_race_track_simulator_value_iteration(True, sys.argv[4], True)
            exit()
        elif sys.argv[2] == "--value" and sys.argv[3] == "--run":
            run_race_track_simulator_value_iteration(True, sys.argv[4], False)
            exit()
        elif sys.argv[2] == "--sarsa":
            run_race_track_simulator_sarsa(True, sys.argv[4])
            exit()

    exit()

