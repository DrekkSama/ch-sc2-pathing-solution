from bot import PathfindingProbe

import asyncio
import logging
import aiohttp

import argparse
import sc2

from sc2.main import run_game
from sc2.data import Race, Difficulty
from sc2.player import Bot




def parse_arguments():
    # Load command line arguments
    parser = argparse.ArgumentParser()

    # Ladder play arguments
    parser.add_argument("--GamePort", type=int, help="Game port.")
    parser.add_argument("--StartPort", type=int, help="Start port.")
    parser.add_argument("--LadderServer", type=str, help="Ladder server.")


    # Local play arguments
    parser.add_argument("--Sc2Version", type=str, help="The version of Starcraft 2 to load.")
    parser.add_argument("--ComputerRace", type=str, default="Protoss",
                        help="Computer race. One of [Terran, Zerg, Protoss, Random]. Default is Terran. Only for local play.")
    parser.add_argument("--ComputerDifficulty", type=str, default="VeryEasy",
                        help=f"Computer difficulty. One of [VeryEasy, Easy, Medium, MediumHard, Hard, Harder, VeryHard, CheatVision, CheatMoney, CheatInsane]. Default is VeryEasy. Only for local play.")
    parser.add_argument("--Map", type=str, default="LightShade_Pathing_1",
                        help="Custom map to use: LightShade_Pathing_0. ")

    # Both Ladder and Local play arguments
    parser.add_argument("--OpponentId", type=str, help="A unique value identifying opponent.")
    parser.add_argument("--Realtime", action='store_true', help="Whether to use realtime mode. Default is false.")

    args, unknown_args = parser.parse_known_args()

    for unknown_arg in unknown_args:
        print(f"Unknown argument: {unknown_arg}")

    # Set the OpponentId if it's not already set
    if args.OpponentId is None:
        if args.LadderServer:
            args.OpponentId = "None"
        else:
            args.OpponentId = f"{args.ComputerRace}_{args.ComputerDifficulty}"

    return args


def load_bot(args):
    # Load bot
    pathfinding_probe = PathfindingProbe()
    # Add opponent_id to the bot class (accessed through self.opponent_id)
    pathfinding_probe.opponent_id = args.OpponentId

    return Bot(PathfindingProbe.RACE, pathfinding_probe)


def run():
    args = parse_arguments()

    bot = load_bot(args)

    print("Starting Challenge...")
    run_game(
        sc2.maps.get(args.Map),
        [bot],  
        realtime=args.Realtime,
        sc2_version=args.Sc2Version,
    )


# Start game
if __name__ == "__main__":
    run()
