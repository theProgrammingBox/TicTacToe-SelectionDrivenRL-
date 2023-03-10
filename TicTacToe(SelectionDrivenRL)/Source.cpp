#include "NeuralNetwork.h"

int main()
{
	constexpr uint32_t BOARD_WIDTH = 3;
	constexpr uint32_t BOARD_SIZE = BOARD_WIDTH * BOARD_WIDTH;
	NeuralNetwork network;
	
	bool* playerOneWins;
	bool* playerTwoWins;
	float board[BOARD_SIZE];
	int row[BOARD_SIZE];
	int col[BOARD_SIZE];
	int diagonal;
	int antiDiagonal;
	uint32_t numMoves;
	float turn;
	bool gameRunning;
	
	//
	for (uint32_t i = 10000000; i--;)
	{
		playerOneWins = new bool(false);
		playerTwoWins = new bool(false);
		memset(board, 0, sizeof(float) * BOARD_SIZE);
		memset(row, 0, sizeof(int) * BOARD_SIZE);
		memset(col, 0, sizeof(int) * BOARD_SIZE);
		diagonal = 0;
		antiDiagonal = 0;
		numMoves = 0;
		turn = 1;
		gameRunning = true;

		//printf("\nNew Game of TicTacToe\n");
		while (gameRunning)
		{
			/*for (uint32_t i = 0; i < BOARD_WIDTH; i++)
			{
				for (uint32_t j = 0; j < BOARD_WIDTH; j++)
				{
					float value = board[i * BOARD_WIDTH + j];
					if (value == -1)
						printf("O");
					else if (value == 1)
						printf("X");
					else
						printf("-");
				}
				printf("\n");
			}
			printf("\n");*/
			
			uint32_t playerInput;
			if (turn == 1)
			{
				playerInput = network.ForwardPropagate(board, turn, playerOneWins);
			}
			else
			{
				playerInput = network.ForwardPropagate(board, turn, playerTwoWins);
			}
			
			if (board[playerInput] != 0)
			{
				gameRunning = false;
				if (turn == 1)
				{
					//printf("Player 2 Wins due to Invalid Input\n");
					*playerTwoWins = true;
				}
				else
				{
					//printf("Player 1 Wins due to Invalid Input\n");
					*playerOneWins = true;
				}
				break;
			}
			board[playerInput] = turn;
			*(int32_t*)&turn ^= 0x80000000;

			uint32_t rowPos = playerInput / BOARD_WIDTH;
			uint32_t colPos = playerInput % BOARD_WIDTH;
			row[rowPos] += turn;
			col[colPos] += turn;
			diagonal += turn * (rowPos == colPos);
			antiDiagonal += turn * (rowPos + colPos + 1 == BOARD_WIDTH);
			numMoves++;

			if (numMoves == BOARD_SIZE)
			{
				gameRunning = false;
				//printf("Draw\n");
				*playerOneWins = false;
				*playerTwoWins = false;
				break;
			}
			else if (row[rowPos] == BOARD_WIDTH || col[colPos] == BOARD_WIDTH || diagonal == BOARD_WIDTH || antiDiagonal == BOARD_WIDTH)
			{
				gameRunning = false;
				//printf("Player 2 Wins\n");
				*playerTwoWins = true;
				break;
			}
			else if (-row[rowPos] == BOARD_WIDTH || -col[colPos] == BOARD_WIDTH || -diagonal == BOARD_WIDTH || -antiDiagonal == BOARD_WIDTH)
			{
				gameRunning = false;
				//printf("Player 1 Wins\n");
				*playerOneWins = true;
				break;
			}
		}
		
		network.BackPropagate(0.1f);
		delete playerOneWins;
		delete playerTwoWins;
	}
	//
	
	//
	playerOneWins = new bool(false);
	playerTwoWins = new bool(false);
	memset(board, 0, sizeof(float)* BOARD_SIZE);
	memset(row, 0, sizeof(int)* BOARD_SIZE);
	memset(col, 0, sizeof(int)* BOARD_SIZE);
	diagonal = 0;
	antiDiagonal = 0;
	numMoves = 0;
	turn = 1;
	gameRunning = true;

	printf("\nNew Game of TicTacToe\n");
	while (gameRunning)
	{
		for (uint32_t i = 0; i < BOARD_WIDTH; i++)
		{
			for (uint32_t j = 0; j < BOARD_WIDTH; j++)
			{
				float value = board[i * BOARD_WIDTH + j];
				if (value == -1)
					printf("O");
				else if (value == 1)
					printf("X");
				else
					printf("-");
			}
			printf("\n");
		}
		printf("\n");

		uint32_t playerInput;
		if (turn == 1)
		{
			playerInput = network.ForwardPropagate(board, turn, playerOneWins);
		}
		else
		{
			playerInput = network.ForwardPropagate(board, turn, playerTwoWins);
		}

		if (board[playerInput] != 0)
		{
			gameRunning = false;
			if (turn == 1)
			{
				printf("Player 2 Wins due to Invalid Input\n");
				*playerTwoWins = true;
			}
			else
			{
				printf("Player 1 Wins due to Invalid Input\n");
				*playerOneWins = true;
			}
			break;
		}
		board[playerInput] = turn;
		*(int32_t*)&turn ^= 0x80000000;

		uint32_t rowPos = playerInput / BOARD_WIDTH;
		uint32_t colPos = playerInput % BOARD_WIDTH;
		row[rowPos] += turn;
		col[colPos] += turn;
		diagonal += turn * (rowPos == colPos);
		antiDiagonal += turn * (rowPos + colPos + 1 == BOARD_WIDTH);
		numMoves++;

		if (numMoves == BOARD_SIZE)
		{
			gameRunning = false;
			printf("Draw\n");
			*playerOneWins = false;
			*playerTwoWins = false;
			break;
		}
		else if (row[rowPos] == BOARD_WIDTH || col[colPos] == BOARD_WIDTH || diagonal == BOARD_WIDTH || antiDiagonal == BOARD_WIDTH)
		{
			gameRunning = false;
			printf("Player 2 Wins\n");
			*playerTwoWins = true;
			break;
		}
		else if (-row[rowPos] == BOARD_WIDTH || -col[colPos] == BOARD_WIDTH || -diagonal == BOARD_WIDTH || -antiDiagonal == BOARD_WIDTH)
		{
			gameRunning = false;
			printf("Player 1 Wins\n");
			*playerOneWins = true;
			break;
		}
	}

	network.BackPropagate(0.1f);
	delete playerOneWins;
	delete playerTwoWins;
	//

	return 0;
}