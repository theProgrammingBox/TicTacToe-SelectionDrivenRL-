#include "Header.h"

int main()
{
	constexpr uint32_t BOARD_WIDTH = 3;
	constexpr uint32_t BOARD_SIZE = BOARD_WIDTH * BOARD_WIDTH;

	// 1x10 input
	// 10x20 weight
	// 1x20 product
	// 1x20 bias
	// 1x20 leaky activation
	// 20x30 weight
	// 1x30 product
	// 1x30 bias
	// 1x30 leaky activation
	// 30x9 weight
	// 1x9 product
	// 1x9 bias
	// 1x9 softmax activation
	
	constexpr uint32_t INPUT_SIZE = BOARD_SIZE + 1;
	constexpr uint32_t LEAKY_ONE_SIZE = 20;
	constexpr uint32_t WEIGHT_ONE_SIZE = INPUT_SIZE * LEAKY_ONE_SIZE;
	constexpr uint32_t LEAKY_TWO_SIZE = 30;
	constexpr uint32_t WEIGHT_TWO_SIZE = LEAKY_ONE_SIZE * LEAKY_TWO_SIZE;
	constexpr uint32_t SOFTMAX_SIZE = 9;
	constexpr uint32_t WEIGHT_THREE_SIZE = LEAKY_TWO_SIZE * SOFTMAX_SIZE;

	float inputMatrix[INPUT_SIZE];
	float weightMatrixOne[WEIGHT_ONE_SIZE];
	float productMatrixOne[LEAKY_ONE_SIZE];
	float biasMatrixOne[LEAKY_ONE_SIZE];
	float leakyMatrixOne[LEAKY_ONE_SIZE];
	float weightMatrixTwo[WEIGHT_TWO_SIZE];
	float productMatrixTwo[LEAKY_TWO_SIZE];
	float biasMatrixTwo[LEAKY_TWO_SIZE];
	float leakyMatrixTwo[LEAKY_TWO_SIZE];
	float weightMatrixThree[WEIGHT_THREE_SIZE];
	float productMatrixThree[SOFTMAX_SIZE];
	float biasMatrixThree[SOFTMAX_SIZE];
	float softmaxMatrix[SOFTMAX_SIZE];

	for (;;)
	{
		printf("\n\nNew Game of TicTacToe\n");
		uint32_t numMoves = 0;
		int row[BOARD_SIZE];
		int col[BOARD_SIZE];
		int diagonal = 0;
		int antiDiagonal = 0;
		int turn = 1;
		int board[BOARD_SIZE];
		memset(row, 0, sizeof(int) * BOARD_SIZE);
		memset(col, 0, sizeof(int) * BOARD_SIZE);
		memset(board, 0, sizeof(int) * BOARD_SIZE);

		bool gameRunning = true;
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

			uint32_t playerInput;
			/*printf("Enter position (0 - 8): ");
			scanf_s("%d", &playerInput);
			if (playerInput < 0 || playerInput > 8 || board[playerInput] != 0)
			{
				gameRunning = false;
				if (turn == 1)
				{
					printf("Player 2 Wins due to Invalid Input\n");
				}
				else
				{
					printf("Player 1 Wins due to Invalid Input\n");
				}
				break;
			}*/
			float input[BOARD_SIZE + 1];
			memcpy(input, board, sizeof(int) * BOARD_SIZE);
			input[BOARD_SIZE] = turn;
			
			board[playerInput] = turn;
			turn *= -1;

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
				break;
			}
			else if (row[rowPos] == BOARD_WIDTH || col[colPos] == BOARD_WIDTH || diagonal == BOARD_WIDTH || antiDiagonal == BOARD_WIDTH)
			{
				gameRunning = false;
				printf("Player 2 Wins\n");
				break;
			}
			else if (-row[rowPos] == BOARD_WIDTH || -col[colPos] == BOARD_WIDTH || -diagonal == BOARD_WIDTH || -antiDiagonal == BOARD_WIDTH)
			{
				gameRunning = false;
				printf("Player 1 Wins\n");
				break;
			}
		}
	}

	return 0;
}