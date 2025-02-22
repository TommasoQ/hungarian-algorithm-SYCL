// Hungarian algorithm in C++

#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>
#include <fstream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <chrono>


// define a constant for infinity
const int inf = std::numeric_limits<int>::max();

using namespace std;


// Function that, given a matrix, subtracts the minimum value of each row from all elements of that row and return the result.

void row_reduction(vector<vector<int>> &matrix) {
    for (int i = 0; i < matrix.size(); i++) {
        int min_val = matrix[i][0];
        for (int j = 0; j < matrix.size(); j++) {
            if (matrix[i][j] < min_val) {
                min_val = matrix[i][j];
            }
        }
        for (int j = 0; j < matrix.size(); j++) {
            matrix[i][j] = matrix[i][j] - min_val;
        }
    }
}

// Function that, given a matrix, subtracts the minimum value of each column from all elements of that column and return the result.
// If the minimum element in the column is 0, then skip to the next column without subracting.
// Do not re-instantiate the matrix, just modify the input matrix.

void column_reduction(vector<vector<int>> &matrix) {
    for (int i = 0; i < matrix.size(); i++) {
        int min_val = matrix[0][i];
        for (int j = 0; j < matrix.size(); j++) {
            if (matrix[j][i] < min_val) {
                min_val = matrix[j][i];
            }
        }
        for (int j = 0; j < matrix.size() && min_val != 0; j++) {
            matrix[j][i] = matrix[j][i] - min_val;
        }
    }
}

// Checking if the optimal assignemt is found or not.
// We will refer to the 0s of the matrix as "stars" and "primes".
// The matrix mask will keep track of the stars and primes, in particular mask(i,j) = 1 if there is a star at (i,j) and mask(i,j) = 2 if there is a prime at (i,j).
// Stars and primes will then be used to build the augmenting path, check if the otpimal assignmet is found,
// and determine the assignment itself.

// Step 1: star the zeros.
// Proceding in order, find a zero in the matrix and check if the row or column was already covered.
// If not, star the zero and cover the row and column.
// If yes, another zero was already starred in the same row or column, then skip to the next zero.
// Repeat until all zeros are starred.
// If at the end all the columns were covered, then the optimal assignment is found.

void starring_the_zeros(vector<vector<int>> &matrix, vector<vector<int>> &mask, vector<int> &row_cover, vector<int> &col_cover) {

    int n = matrix.size();

    fill(col_cover.begin(), col_cover.end(), 0);
    fill(row_cover.begin(), row_cover.end(), 0);


    // Reset the mask to all 0s
    for (int i = 0; i < n; i++) {
        fill(mask[i].begin(), mask[i].end(), 0);
    }

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){  
            if (matrix[i][j] == 0){
                if (row_cover[i] == 0 && col_cover[j] == 0){
                    mask[i][j] = 1;
                    row_cover[i] = 1;
                    col_cover[j] = 1;
                    break;
                }
            }         
        }

    }

    /*// Reset the row and column cover to all 0s
    fill(row_cover.begin(), row_cover.end(), 0);    
    fill(col_cover.begin(), col_cover.end(), 0);
    */


}

// Check if all columns are covered.

void all_columns_covered(vector<vector<int>> &mask, vector<int> &col_cover, vector<int> &row_cover, bool &found){
    int n = mask.size();
    int col_count = 0;

    // Reset the row and column cover to all 0s
    fill(row_cover.begin(), row_cover.end(), 0);
    fill(col_cover.begin(), col_cover.end(), 0);

    // Cover the columns with a star in the mask
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (mask[j][i] == 1){
                col_cover[i] = 1;
            }
        }
    }

    for (auto& n: col_cover)
        if (n == 1)
            col_count++;
    
    if (col_count >= n) {
        found = true; // solution found
    }
    else {
        found = false; // solution not found
    }
}

// Functions used following Step 2 of the algorithm.

// Find the position of an uncovered zero in the matrix.
// If there is no uncovered zero, return -1 as the coordinates.

void find_uncovered_zero(int &row, int &col, vector<vector<int>> &matrix, vector<int> &row_cover, vector<int> &col_cover) {
    int n = matrix.size();
    bool done = false;
    row = -1;
    col = -1;
    int i = 0;
    int j = 0;

    for (int i = 0; i < n; ++i) {
        if (done) {
            break;
        }
        for (int j = 0; j < n; ++j) {
            if (matrix[i][j] == 0 && row_cover[i] == 0 && col_cover[j] == 0) {
                row = i;
                col = j;
                done = true;
                break;
            }
        }
    }

}

// Check if there is already a star in the row.

bool there_is_star_in_row(int &row, int& col, vector<vector<int>> &mask){
    int n = mask.size();
    for (int i = 0; i < n; i++){
        if (mask[row][i] == 1){
            col = i;
            return true;
        }
    }
    return false;
}

// Find the star in the row.

void find_star_in_row(int row, vector<vector<int>> &mask, int &col){
    int n = mask.size();
    for (int i = 0; i < n; i++){
        if (mask[row][i] == 1){
            col = i;
        }
    }
}

// Step 2: Find a non-covered zero and prime it.
// If there is no star in the same row, then we will proced to find an augmenting path.
// If there is a star in the same row, then we will cover the row and uncover the column of the star.
// Repeat until all zeros are covered.

void find_prime_and_uncover_star(vector<vector<int>> &matrix, vector<vector<int>> &mask, vector<int> &row_cover, 
                                vector<int> &col_cover, int &path_row_0, int &path_col_0, bool &optimize){
    int n = matrix.size();
    int row;
    int col;
    bool done = false;

    // Print the mask
    /*cout << "Mask at the beginning of find prime and uncover star: " << endl;
    for (int i = 0; i < mask.size(); i++) {
        for (int j = 0; j < mask.size(); j++) {
            cout << mask[i][j] << " ";
        }
        cout << endl;
    }*/

    while (!done){
        find_uncovered_zero(row, col, matrix, row_cover, col_cover);
        //cout << "I'm inside find_prime_and_uncover_star..." << endl;
        //cout << "Row: " << row << " Col: " << col << endl;

        if (row == -1){
            optimize = true;
            break;
        }
        else {
            mask[row][col] = 2;
            if (there_is_star_in_row(row, col, mask)) {
                //find_star_in_row(row, mask, col);
                row_cover[row] = 1;
                col_cover[col] = 0;
            }
            else {
                done = true;
                path_row_0 = row;
                path_col_0 = col;
                optimize = false;
            }
        }
    }

    // Print the mask
    /*cout << "Mask at the end of find prime and uncover star: " << endl;
    for (int i = 0; i < mask.size(); i++) {
        for (int j = 0; j < mask.size(); j++) {
            cout << mask[i][j] << " ";
        }
        cout << endl;
    }*/

}

// Functions used in following Step 3 of the algorithm.

// Find star in the column.

void find_star_in_col(int &col, vector<vector<int>> &mask, int &row){
    int n = mask.size();
    for (int i = 0; i < n ; i++){
        if (/*i != row &&*/ mask[i][col] == 1){
                row = i;
                break;
        }
        else {
            row = -1;
        }
    }
    //cout << "Found star in row: " << row << endl;
}

// Find prime in the row.

void find_prime_in_row(int &row, vector<vector<int>> &mask, int &col){
    int n = mask.size();
    for (int i = 0; i < n; i++){
        if (mask[row][i] == 2){
            col = i;
        }
    }
    //cout << "Found prime in column: " << col << endl;
}

// Augmenting path.

void augment_path(vector<vector<int>> &path, int &path_count, vector<vector<int>> &mask){
    for (int p = 0; p < path_count; p++){
        if (mask[path[p][0]][path[p][1]] == 1)
            mask[path[p][0]][path[p][1]] = 0;
        else
            mask[path[p][0]][path[p][1]] = 1;
    }
}

// Step 3: Construct the alternating series of primes and stars.
// Start from the uncovered primed zero found before:
// Find the star in the same column as the prime.
// Find the prime in the same row as the star.
// Repeat until the series terminates with a prime that has no star in the same column.
// Unstar all the stars of the series and star all the primes of the series.
// Erase all primes and uncover every line.
// Go back to checking if all the columns are covered.

void alternating_path(vector<vector<int>>& path, int &path_row_0, int &path_col_0, vector<vector<int>>& mask, vector<int>& row_cover,
                        vector<int>& col_cover){
    int row = -1;
    int col = -1;
    int path_count = 1;

    // Reset the matrix path to all 0s
    for (int i = 0; i < path.size(); i++) {
        for (int j = 0; j < 2; j++){
            path[i][j] = 0;
        }
    }
    
    path[path_count - 1][0] = path_row_0;
    path[path_count - 1][1] = path_col_0;
    
    bool done = false;
    while(!done){
        find_star_in_col(path[path_count - 1][1], mask, row);
        if (row > -1) {
            
            path_count += 1;
            path[path_count - 1][0] = row;
            path[path_count - 1][1] = path[path_count - 2][1];
            
        }
        else {done = true;}
        
        if (!done) {
            find_prime_in_row(path[path_count - 1][0], mask, col);
            path_count += 1;
            path[path_count - 1][0] = path[path_count - 2][0];
            path[path_count - 1][1] = col;
            

        }
    }
    
    // print path
    /*
    cout << "Path: " << endl;
    for (int i = 0; i < path.size(); i++) {
        for (int j = 0; j < 2; j++) {
            cout << path[i][j] << " ";
        }
        cout << endl;
    }*/

    augment_path(path, path_count, mask);

    // print mask

    /*cout << "Mask after augment_path: " << endl;
    for (int i = 0; i < mask.size(); i++) {
        for (int j = 0; j < mask.size(); j++) {
            cout << mask[i][j] << " ";
        }
        cout << endl;
    }*/

    // print row cover and col cover

    /*cout << "Row cover after augment_path: " << endl;
    for (int i = 0; i < row_cover.size(); i++) {
        cout << row_cover[i] << " ";
    }
    cout << endl;

    cout << "Col cover after augment_path: " << endl;
    for (int i = 0; i < col_cover.size(); i++) {
        cout << col_cover[i] << " ";
    }
    cout << endl;
    */
    /* Reset the row and column cover to all 0s
    fill(row_cover.begin(), row_cover.end(), 0);
    fill(col_cover.begin(), col_cover.end(), 0);
    */

    // Erase all primes
    for (auto& r: mask){
        for (auto& val: r){
            if (val == 2)
                val = 0;
        }
    }

    // print mask

    /*cout << "Mask after erasing all primes: " << endl;
    for (int i = 0; i < mask.size(); i++) {
        for (int j = 0; j < mask.size(); j++) {
            cout << mask[i][j] << " ";
        }
        cout << endl;
    }*/
}

// Functions useful for step 4.

// Draw the lines that cover all the zeros of matrix.

void draw_lines(vector<int> &row_cover, vector<int> &col_cover, vector<vector<int>> &lines) {
    int n = row_cover.size();

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if ((row_cover[i] == 1 && col_cover[j] == 0) || ( row_cover[i] == 0 && col_cover[j] == 1)){
                lines[i][j] = 1;
            }
            else if (row_cover[i] == 1 && col_cover[j] == 1){
                lines[i][j] = 2;
            }
        }
    }
}

// Step 4: Step towards optimality.
// Find the smallest uncovered element in the matrix.
// Subtract it from all uncovered elements.
// Add it to all elements covered by two lines.

void step_towards_optimality(vector<vector<int>> &matrix, vector<int> &row_cover, vector<int> &col_cover) {
    int n = matrix.size();
    int min_val = inf;

    // Find the minimum value of the elements that are not crossed by the lines
    for (int i = 0; i < n; i++){
        if (row_cover[i] == 0){
            for (int j = 0; j < n; j++){
                if (col_cover[j] == 0 && matrix[i][j] < min_val){
                    min_val = matrix[i][j];
                }
            }
        }
    }

    // Subtract the minimum value from the elements that are not crossed by the lines and add it to the elements that are crossed by two lines
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (row_cover[i] == 0 && col_cover[j] == 0){
                matrix[i][j] -= min_val;
            }
            else if (row_cover[i] == 1 && col_cover[j] == 1){
                matrix[i][j] += min_val;
            }
        }
    }
}

// Calculate the cost of the optimal assignment.

vector<pair<int,int>> optimal_assignment(vector<vector<int>> &matrix, vector<vector<int>> &mask)
{
    int n = matrix.size();
    vector<pair<int, int>> assignment;
    int total_cost = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (mask[i][j] == 1)
            {
                assignment.push_back(make_pair(i, j));
                total_cost += matrix[i][j];
            }
        }
    }

    /*std::cout << "Optimal assignment: ";
    for (int i = 0; i < assignment.size(); i++)
    {
        std::cout << "(" << assignment[i].first << ", " << assignment[i].second << ") ";
    }
    std::cout << std::endl;*/
    std::cout << "Total cost: " << total_cost << std::endl;
    
    return assignment;
}

vector<vector<int>> read_matrix(string filename) {
    ifstream file(filename);

    vector<vector<int>> matrix;
    vector<int> list;
    int n;
    int count;

    if (!file.is_open()){
        cout << "File not found!" << endl;
    }
    else {
        while (file >> n){
            list.push_back(n);

        }

        count = sqrt(list.size());

        for (int i = 0; i < count; i++){
            vector<int> row;
            for (int j = 0; j < count; j++){
                row.push_back(list[i*count + j]);
            }
            matrix.push_back(row);
        }
    }

    file.close();
    /*
    for (int i = 0; i < count; i++){
        matrix[i][i] = inf;
    }*/
    
    return matrix;
}


// Main

int main(){

    vector<vector<int>> matrix = read_matrix("assign800.txt");

    // Initialize the variables
	
    int size = matrix.size();
    vector<vector<int>> M(matrix);
    vector<vector<int>> mask(size, vector<int>(size, 0));
    vector<int> row_cover(size, 0);
    vector<int> col_cover(size, 0);
    vector<vector<int>> lines(size, vector<int>(size, 0));
    bool found = false;
    int path_row_0;
    int path_col_0;
    vector<pair<int,int>> assignment;
    bool TSP = false;
    bool optimize;

    // Array for the augmenting path algorithm
    std::vector<std::vector<int>> path (size+10, std::vector<int>(2, 0));
	
    cout << "Starting hungarian algorithm..." << endl;

	auto start = std::chrono::high_resolution_clock::now();

    
    // Preprocessing the matrix.

    row_reduction(M);

    column_reduction(M);

    // Star the zeros
    starring_the_zeros(M, mask, row_cover, col_cover);

    // Proceding with the algorithm.
    while(!found){
        all_columns_covered(mask, col_cover, row_cover, found);
        if (found){
            break;
        }
        else {
            find_prime_and_uncover_star(M, mask, row_cover, col_cover, path_row_0, path_col_0, optimize);

            if (optimize){
                step_towards_optimality(M, row_cover, col_cover);
                
                // Star the zeros
                starring_the_zeros(M, mask, row_cover, col_cover);
            }
            else {
                alternating_path(path, path_row_0, path_col_0, mask, row_cover, col_cover);
            }
        }
    }

    // Calculate the optimal assignment.
    
    assignment = optimal_assignment(matrix, mask);  
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Time elapsed: " << (end-start).count()/1e6 << " milliseconds" << endl;
    
    return 0;
}