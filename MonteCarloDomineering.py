import numpy as np
import random
from random import randint

def coup_un(A,i,j):
    if A[i,j] == 0 and A[i+1,j] == 0:
        A[i,j] = 1
        A[i+1,j] = 1

def coup_deux(A,i,j):
    if A[i,j] == 0 and A[i,j+1] == 0:
        A[i,j] = 2
        A[i,j+1] = 2

def coups_possibles_un(A):
    liste_possible = []
    for i in range(0,len(A)-1):
        for j in range(0,len(A)):
            if A[i,j] ==0 and A[i+1,j] == 0:
                liste_possible.append((i,j))
    return liste_possible

def coups_possibles_deux(A):
    liste_possible = []
    for i in range(0,len(A)):
        for j in range(0,len(A)-1):
            if A[i,j] ==0 and A[i,j+1] == 0:
                liste_possible.append((i,j))
    return liste_possible

def rand_domineering(A, player):
    while 1:
        if player == 1:
            if coups_possibles_un(A) != []:
                play = random.choice(coups_possibles_un(A))
                coup_un(A, play[0], play[1])
                player = 2
            else:
                return player
        if player == 2:
            if coups_possibles_deux(A) != []:
                play = random.choice(coups_possibles_deux(A))
                coup_deux(A, play[0], play[1])
                player = 1
            else:
                return player

def mc_domineering(A, player):
    while 1:
        if player == 1:
            if coups_possibles_un(A) != []:
                play = montecarlo(A, player)
                coup_un(A, play[0], play[1])
                player = 2
            else:
                return player
        if player == 2:
            if coups_possibles_deux(A) != []:
                play = montecarlo(A, player)
                coup_deux(A, play[0], play[1])
                player = 1
            else:
                return player

def montecarlo(A, player):
    B = A[:]
    if player == 1:
        moves = coups_possibles_un(A)
        if moves != []:
            for coup in moves:
                victoire = np.zeros(len(moves))
                for i in range(0,20):
                    coup_un(A, coup[0], coup[1])
                    loser = rand_domineering(A, 2)
                    if loser == 2:
                        victoire[moves.index(coup)] = victoire[moves.index(coup)] + 1
                    A = B[:]
    if player == 2:
        moves = coups_possibles_deux(A)
        if moves != []:
            for coup in moves:
                victoire = np.zeros(len(moves))
                for i in range(0,20):
                    coup_deux(A, coup[0], coup[1])
                    loser = rand_domineering(A, 1)
                    if loser == 1:
                        victoire[moves.index(coup)] = victoire[moves.index(coup)] + 1
                    A = B[:]
    victoire = victoire / 20
    max = 0
    for i in range(0, len(victoire)):
        if victoire[i] > max:
            max = victoire[i]
            coupchoisi = moves[i]
    return coupchoisi

def rand_vs_mc_domineering(A, player):
    while 1:
        if player == 1:
            if coups_possibles_un(A) != []:
                play = montecarlo(A, player)
                coup_un(A, play[0], play[1])
                player = 2
            else:
                return player
        if player == 2:
            if coups_possibles_deux(A) != []:
                play = random.choice(coups_possibles_deux(A))
                coup_deux(A, play[0], play[1])
                player = 1
            else:
                return player

losers_ran = []
losers_mc = []
losers_mc_vs_mc = []

for i in range(0,100):
    player = randint(1,2)
    damier = np.zeros((8, 8))
    losers_ran.append(rand_domineering(damier, player))

for i in range(0,100):
    player = randint(1,2)
    damier = np.zeros((8, 8))
    losers_mc.append(mc_domineering(damier, player))

for i in range(0, 100):
    player = randint(1, 2)
    damier = np.zeros((8, 8))
    losers_mc_vs_mc.append(rand_vs_mc_domineering(damier, player))

print("In Random Domineering, Player 1 loses (%) : ", losers_ran.count(1)/100)
print("In Monte Carlo Domineering, Player 1 loses (%) : ", losers_mc.count(1)/100)
print("In Monte Carlo (Player 1) vas Random (Player 2) Domineering, Player 1 loses : (%) ", losers_mc_vs_mc.count(1)/100)