from game import gameState, integerInput
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from colorama import Fore
print("imports done")

def fillBoard(input):
    grid = copy.deepcopy(input)
    for x in range(0,len(grid)):
        for y in range(0,len(grid[x])):
            grid[x][y] = (2 * grid[x][y]) - 1
        grid[x] = grid[x] + ([0] * (6 - len(grid[x])))
    return grid

def gameStateToTensor(input):
    grid = fillBoard(input.grid)
    turn = (input.turn * 2) - 1
    grid = torch.tensor(grid)
    grid = grid.view(-1)
    return (torch.cat((grid,torch.tensor([turn])))).float()

class AIEvaluationFunction(torch.nn.Module):
    def __init__(self):
        super(AIEvaluationFunction,self).__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(43,96))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(96,96))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(96,1))
        self.layers.append(torch.nn.Tanh())
        
        self.parameterList = nn.ParameterList()

        for parameter in self.layers:
            self.parameterList.append(parameter)

    def forward(self,input):
        if type(input) != type(torch.tensor([])):
            neuronStates = gameStateToTensor(input)
        else:
            neuronStates = input

        for state in neuronStates:
            state = (state * 2) - 1

        for layer in self.layers:
            neuronStates = layer(neuronStates)
        
        return neuronStates

def evaluatePosition(state, alpha, beta, turn, depth, maxDepth, AI, monitor):
    winner = state.winCheck()
    if winner is not None:
        if monitor != -1:
            monitor += 1
            print(str(monitor) + "/" + str(7**(maxDepth + 1)))
        if winner is not False:
            return [(2 * winner) - 1, None, [beta, alpha][turn]]
        return [0, None, [beta, alpha][turn]]

    if depth < maxDepth:
        childValues = []
        childAlpha = alpha
        childBeta = beta

        for moveNumber in range(0, 7):
            newState = copy.deepcopy(state)
            valid = newState.move(moveNumber)
            if valid:
                output = evaluatePosition(newState, childAlpha, childBeta, not turn, depth + 1, maxDepth, AI, [monitor, -1][monitor == -1])

                value = output[0]
                childValues.append(value)
                if turn:
                    childAlpha = max(childAlpha, value)
                else:
                    childBeta = min(childBeta, value)

                if childBeta <= childAlpha:
                    break
            else:
                childValues.append([float("inf"), float("-inf")][turn])
            if monitor != -1:
                monitor += (7**(maxDepth - depth))
        if monitor != -1:
            print(str(monitor) + "/" + str(7**(maxDepth + 1)))
        return [[min(childValues), max(childValues)][turn], [childValues.index(min(childValues)), childValues.index(max(childValues))][turn], [childBeta, childAlpha][turn]]

    else:
        #We have reached maximum depth, time to use the AI
        output = AI.forward(state)
        if monitor != -1:
            monitor += 1
            print(str(monitor)+"/"+str(7**(maxDepth+1)))
            print(output.item())
        return [output.item(),None,[beta,alpha][turn]]
        #Wow, that was easier than I thought
        #Later comment, wow, this is much harder than I thought

def getBestMove(state,player,depth,AI,monitor=True):#Player is flase for minimizer, true for maximizer
    return evaluatePosition(state,float("-infinity"),float("infinity"),player,0,depth,AI,[-1,0][monitor])[1]

def selfPlayTrain(url,numOfEpochs,depth):
    print(Fore.RESET + "",end="")
    #Load model from json file
    model = AIEvaluationFunction()
    model_state_dict = torch.load(url)
    model.load_state_dict(model_state_dict)

    

    for epoch in range(0,numOfEpochs):
        #List of game states the player has been in
        data = torch.tensor([])

        winner = None
        game = gameState()
        while winner == None:
            #Get the current grid, including the empty squares, add it to the data
            grid = fillBoard(game.grid)
            data = torch.cat((data,gameStateToTensor(game)))

            #Get best move
            move = getBestMove(game,[False,True][game.turn],depth,model,False)

            game.move(move)

            game.display()

            winner = game.winCheck()

        data = data.view(len(data) // 43,43)#Reshape data tensor
        if str(type(winner)) == "<class 'int'>":
            #Sort the good data from the bad data

            if winner == 0:#Yellow won
                print(Fore.YELLOW + "Yellow won!")
            
            else:#Red won
                print(Fore.RED + "Red won!")
            print(Fore.RESET + "",end="")
            gameValue = (2 * winner) - 1

        else:
            print(Fore.RESET + "Tie game!")
            gameValue = 0
        
        dataSet = TensorDataset(data,torch.tensor([gameValue] * len(data)))
        dataLoader = DataLoader(dataSet,batch_size=len(data),shuffle=True)
        optimizer = optim.SGD(model.parameters(),lr=.01)
        lossFunc = nn.MSELoss()

        for i, (batchStates,batchValues) in enumerate(dataLoader):
            optimizer.zero_grad()
            output = model.forward(batchStates)
            loss = lossFunc(output.float(), torch.tensor(batchValues).float())
            print(Fore.CYAN + str(loss))
            print(Fore.RESET + "",end="")
            loss.backward()
            optimizer.step()

        print(Fore.GREEN + "EPOCH:" + str(epoch + 1) + "/" + str(numOfEpochs))
    #Save model to json
    torch.save(model.state_dict(), url)
    


def humanVAI(depth,monitor=True,humanTurn=True):
    b = gameState()

    model = AIEvaluationFunction()
    model_state_dict = torch.load("currentModel.json")
    model.load_state_dict(model_state_dict)

    b.display()
    while b.winCheck() == None:
        if humanTurn:
            while not b.move(integerInput(f"Human to move:\n",0,6)):
                pass
        else:
            b.move(getBestMove(b,[False,True][b.turn],depth,model,monitor))
        b.display()
        humanTurn = not humanTurn
    if b.winCheck == False:
        print("Tie game!")
    else:
        print(["Human","AI"][humanTurn] + " wins!")

def initialize():
    model = AIEvaluationFunction()
    torch.save(model.state_dict(),"currentModel.json")
    torch.save(model.state_dict(),"untrainedModel.json")

if __name__ == "__main__":
    selfPlayTrain("currentModel.json",1,6)
    #humanVAI(6,False)