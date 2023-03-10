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
	uint32_t player1WinByInvalidMove = 0;
	uint32_t player2WinByInvalidMove = 0;
	uint32_t player1Win = 0;
	uint32_t player2Win = 0;
	uint32_t draw = 0;
	
	for (uint32_t i = 1000000000; i--;)
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

		if (i % 100000 == 0)
		{
			printf("\nStats:\n");
			printf("Player 1 Win By Invalid Move: %d\n", player1WinByInvalidMove);
			printf("Player 2 Win By Invalid Move: %d\n", player2WinByInvalidMove);
			printf("Player 1 Win: %d\n", player1Win);
			printf("Player 2 Win: %d\n", player2Win);
			printf("Draw: %d\n", draw);
			
			player1WinByInvalidMove = 0;
			player2WinByInvalidMove = 0;
			player1Win = 0;
			player2Win = 0;
			draw = 0;
			
			printf("New Game of TicTacToe\n");
		}
		while (gameRunning)
		{

			if (i % 100000 == 0)
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
			}
			
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
					player2WinByInvalidMove++;
					*playerTwoWins = true;
				}
				else
				{
					//printf("Player 1 Wins due to Invalid Input\n");
					player1WinByInvalidMove++;
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
				draw++;
				break;
			}
			else if (row[rowPos] == BOARD_WIDTH || col[colPos] == BOARD_WIDTH || diagonal == BOARD_WIDTH || antiDiagonal == BOARD_WIDTH)
			{
				gameRunning = false;
				//printf("Player 2 Wins\n");
				player2Win++;
				*playerTwoWins = true;
				break;
			}
			else if (-row[rowPos] == BOARD_WIDTH || -col[colPos] == BOARD_WIDTH || -diagonal == BOARD_WIDTH || -antiDiagonal == BOARD_WIDTH)
			{
				gameRunning = false;
				//printf("Player 1 Wins\n");
				player1Win++;
				*playerOneWins = true;
				break;
			}
		}
		
		network.BackPropagate(1.0f);
		delete playerOneWins;
		delete playerTwoWins;
	}

	return 0;
}