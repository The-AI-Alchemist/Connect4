from colorama import Fore

def integerInput(prompt,lowerBound=float("-infinity"),upperBound=float("infinity")):
    while True:
        response = input(prompt)
        try:
            response = int(response)
            if response >= lowerBound and response <= upperBound:
                return response
            print("Please select a number between " + str(lowerBound) + " and " + str(upperBound) + "(inclusive)")
        except:
            print("Not a number, try again")

class gameState:
    def __init__(self):
        self.grid = [[],[],[],[],[],[],[]]
        self.turn = 0
        #Each list is a collumn, the last element in the list is on top

        #Remember, max height is six
    
    def winCheck(self):
        #Check for vertical win
        for collumn in self.grid:
            for i in range(len(collumn)-2):
                if collumn[i:i+4] == [0, 0, 0, 0] or collumn[i:i+4] == [1, 1, 1, 1]:
                    return collumn[i]
        
        #Check for horizontal win
        transposed_grid = []
        max_col_len = max(len(col) for col in self.grid)
        for j in range(max_col_len):
            new_row = []
            for i in range(len(self.grid)):
                if j < len(self.grid[i]):
                    new_row.append(self.grid[i][j])
                else:
                    new_row.append(None)
            transposed_grid.append(new_row)
        
        for collumn in transposed_grid:
            for i in range(len(collumn)-2):
                if collumn[i:i+4] == [0, 0, 0, 0] or collumn[i:i+4] == [1, 1, 1, 1]:
                    return collumn[i]
        
        #Check for diagonal win
        for i in range(len(self.grid)-3):
            for j in range(0,7):
                try:
                    if self.grid[i][j] == self.grid[i+1][j+1] == self.grid[i+2][j+2] == self.grid[i+3][j+3]:
                        return self.grid[i][j]
                except:
                    pass
        
        for i in range(3, len(self.grid)):
            for j in range(0,7):
                try:
                    if self.grid[i][j] == self.grid[i-1][j+1] == self.grid[i-2][j+2] == self.grid[i-3][j+3]:
                        return self.grid[i][j]
                except:
                    pass

        for collumn in self.grid:
            if len(collumn) != 6:
                return None
        return False
        

    def move(self,collumn):
        #Adds a token to the board, and as an output, returns whether the move was valid, invalid moves yield no response
        if len(self.grid[collumn]) < 6:
            self.grid[collumn].append(self.turn)
            self.turn = 1-self.turn
            return True
        return False

    def display(self):
        display_rows = []
        for y in range(0,6):
            row = Fore.RESET + "|"
            for collumn in self.grid:
                try:
                    row = row + {0:Fore.YELLOW + "O",1:Fore.RED + "O"}[collumn[y]]
                except:
                    row = row +Fore.RESET + "-"
            row = row + (Fore.RESET + "|")
            display_rows.append(row)
        display_rows.reverse()
        for row in display_rows:
            print(row)
        if self.turn == 0:
            print(Fore.YELLOW + "-0123456-")
        else:
            print(Fore.RED + "-0123456-")


def humanVhuman():
    b = gameState()
    while True:
        b.display()
        In = integerInput("",0,6)
        try:
            int(In)
        except:
            break
        b.move(int(In))
        winner = b.winCheck()
        if winner != None:
            b.display()
            print([Fore.YELLOW + "Yellow",Fore.RED + "Red"][winner] + " won")
            return None
    print(b.grid)

if __name__ == "__main__":
    integerInput("Test:")