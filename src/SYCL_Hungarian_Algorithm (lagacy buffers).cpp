#include <sycl/sycl.hpp>
using namespace sycl;
using namespace std;
    
constexpr int N=5;
constexpr int SUB_GROUP_SIZE_8=8;
constexpr int SUB_GROUP_SIZE_16=16;
constexpr int SUB_GROUP_SIZE_32=32;
const sycl::property_list props = {sycl::property::buffer::use_host_ptr {}};



void row_reduction (buffer<int, 2> &buf_MATRIX, queue &Q) {
    //const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

    Q.submit([&] (handler &h){
        //auto out = stream(12500, 768, h);  //output stream from device

        accessor MATRIX(buf_MATRIX, h, read_write);
        
        auto R = nd_range<2>(range<2>(N, N), range<2>(1, N)); //one workgroup for each row
        h.parallel_for(R, [=] (nd_item<2> it) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t row = it.get_global_id()[0];
            size_t col = it.get_global_id()[1];

            auto grp = it.get_group();
            MATRIX[row][col] -= reduce_over_group(grp, MATRIX[row][col], minimum<>() );
        });
    }).wait();
}

void column_reduction (buffer<int, 2> &buf_MATRIX, queue &Q) {
    //const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

    Q.submit([&] (handler &h){
        accessor MATRIX(buf_MATRIX, h, read_write);
        
        auto R = nd_range<2>(range<2>(N, N), range<2>(N, 1)); //one workgroup for each column
        h.parallel_for(R, [=] (nd_item<2> it) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t row = it.get_global_id()[0];
            size_t col = it.get_global_id()[1];

            auto grp = it.get_group();
            //TODO: think about letting only one cell compute the min, then broadcast it to the others
            MATRIX[row][col] -= reduce_over_group(grp, MATRIX[row][col], minimum<>() );
        });
    }).wait();
}



void starring_the_zeroes(buffer<int, 2> &buf_MATR, buffer<int, 2> &buf_MASK, buffer<bool, 1> &buf_COL, queue &Q) {

    //reset the mask to 0
    Q.submit([&] (handler &h){
        accessor MASK(buf_MASK, h, write_only, no_init);
        accessor COL(buf_COL, h, write_only, no_init);
        
        auto R = nd_range<2>(range<2>(N, N), range<2>(1, N));
        h.parallel_for(R, [=] (nd_item<2> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t r = item.get_global_id()[0];
            size_t c = item.get_global_id()[1];
            
            MASK[r][c] = 0;
            
            if (r == 0) { // only the first work-group is responsible for resetting col_cover
                COL[c] = 0;
            }
        });
    }).wait();

    
    //star the first zero per row, except the ones excluded by col_cover
    for (int r=0; r<N; r++) {
        Q.submit([&] (handler &h){

            accessor MASK(buf_MASK, h, read_write);
            accessor MATR(buf_MATR, h, read_only);
            accessor COL(buf_COL, h, read_write);

            h.single_task([=] (){
                for (int c = 0; c < N; c++) {
                    if (MATR[r][c]==0 && COL[c]==0) {
                        MASK[r][c] = 1;
                        COL[c] = 1;
                        break;
                    }
                }
            });
        }).wait();
    }
};



//This function resets row_cover and col_cover, then sets col_cover[i] to 1 if the corresponding i column in mask contains a 1 (a star).
//Finally, it checks whether all columns are covered (found=true in that case)
void all_columns_covered(buffer<int, 2> &buf_MASK, buffer<bool, 1> &buf_ROW, buffer<bool, 1> &buf_COL, buffer<bool, 1> &buf_FOUND, queue &Q) {

    //reset row_cover and col_cover to 0
    Q.submit([&] (handler &h){
        accessor ROW(buf_ROW, h, write_only, no_init);
        accessor COL(buf_COL, h, write_only, no_init);
        
        auto R = nd_range<1>(range<1>(N), range<1>(N));
        h.parallel_for(R, [=] (nd_item<1> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            auto idx = item.get_local_id();
            ROW[idx] = 0;
            COL[idx] = 0;
       });
    }).wait();

    //sets col_cover according to the position of stars (1s) on the mask; 
    Q.submit([&] (handler &h) {
        accessor MASK (buf_MASK, h, read_only);
        accessor COL (buf_COL, h, write_only);
        accessor FOUND (buf_FOUND, h, write_only);
        
        auto R = nd_range<2>(range<2>(N, N), range<2>(1, N));
        h.parallel_for(R, [=] (nd_item<2> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t r = item.get_global_id()[0];
            size_t c = item.get_global_id()[1];
            
            if (MASK[r][c]==1) {
                COL[c] = 1;
            }

            auto grp = item.get_group();
            group_barrier(grp);

            if (r==0) { //only the first work-group is needed to check if all coloumns are covered
                FOUND[0] = all_of_group(grp, COL[c]==1);
            }
        });
    }).wait();
    
    /* //TODO: remove this if the above works
    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h);
        
        accessor COL (buf_COL, h, read_only);
        accessor FOUND (buf_FOUND, h, read_write);
        
        auto R = nd_range<1>(range<1>(N), range<1>(N));
        h.parallel_for(R, [=] (nd_item<1> item){
            size_t idx = item.get_global_id()[0];
            auto grp = item.get_group();

            FOUND[0] = all_of_group(grp, COL[idx]==1);
        });
    }).wait();
    */
}



//Protypes to be used in find_prime_and_uncover_star
void find_uncovered_zero(buffer<int, 1> &buf_ROW, buffer<int, 1> &buf_COL, buffer<int, 2> &buf_MATRIX, buffer<bool, 1> &buf_ROW_COVER, buffer<bool, 1> &buf_COL_COVER, queue &Q);
bool there_is_star_in_row(buffer<int, 1> &buf_ROW, buffer<int, 1> buf_COL, buffer<int, 2> &buf_MASK, queue &Q);

//This function starts a loop that finds uncovered zeroes in order (left->right, up->down); if none are found it goes to the optimization step.
//Otherwise it primes the first zero found, then looks for a star in the same row: if there is one then it updates row/col_cover accordingly, and looks for the next uncovered zero.
//If the star is not found it sets path_row/col_0, and exits going to the alternating path (optimize=false).
void find_prime_and_uncover_star(buffer<int, 2> &buf_MATRIX, buffer<int, 2> &buf_MASK, buffer<bool, 1> &buf_ROW_COVER, buffer<bool, 1> &buf_COL_COVER,
                                 buffer<int, 1> &buf_PATH_ROW_0, buffer<int, 1> &buf_PATH_COL_0, bool &optimize, queue &Q){
    
    int row[1];
    int col[1];
    bool done = false;
    bool found_star;

    buffer<int, 1> buf_ROW{&row[0], range<1>(1), props};
    buffer<int, 1> buf_COL{&col[0], range<1>(1), props};

    host_accessor ROW(buf_ROW, read_only);
    host_accessor COL(buf_COL, read_only);
    host_accessor ROW_COVER(buf_ROW_COVER, write_only);
    host_accessor COL_COVER(buf_COL_COVER, write_only);
    host_accessor PATH_ROW_0(buf_PATH_ROW_0, write_only);
    host_accessor PATH_COL_0(buf_PATH_COL_0, write_only);
    host_accessor MASK(buf_MASK, write_only);

    while (!done){
            
        //row and col are set to N, because it is bigger than the max index of the matrix;
        //this is used to check whether find_uncovered_zero failed in finding a prime.
        row[0]=N;
        col[0]=N;

        //find the first uncovered zero (the first prime zero)
        find_uncovered_zero(buf_ROW, buf_COL, buf_MATRIX, buf_ROW_COVER, buf_COL_COVER, Q);
        //TODO erase debug print
        //cout << "I'm inside find_prime_and_uncover_star..." << std::endl;
        //cout << "Found zero prime in [" << row[0] << "," << col[0] << "]\n";


        //if no zero primes are found, go to optimization step
        if (ROW[0] == N) {
            optimize = true;
            done = true;
        }
        else {
            MASK[ROW[0]][COL[0]] = 2;
            
            found_star = there_is_star_in_row(buf_ROW, buf_COL, buf_MASK, Q); //this function also updates col with the position of the star, if found
            if (found_star) {
                ROW_COVER[ROW[0]] = 1;
                COL_COVER[COL[0]] = 0;
            }
            else {
                PATH_ROW_0[0] = ROW[0];
                PATH_COL_0[0] = COL[0];
                optimize = false;
                done = true;
            }
        }
    }
}

void find_uncovered_zero(buffer <int, 1> &buf_ROW, buffer <int, 1> &buf_COL, buffer<int, 2> &buf_MATRIX, buffer<bool, 1> &buf_ROW_COVER, buffer<bool, 1> &buf_COL_COVER, queue &Q) {
    
    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h);
        
        accessor MATR (buf_MATRIX, h, read_only);
        accessor ROW_C (buf_ROW_COVER, h, read_only);
        accessor COL_C (buf_COL_COVER, h, read_only);
        accessor ROW (buf_ROW, h, read_write);
        accessor COL (buf_COL, h, read_write);
        
        auto R = nd_range<2>(range<2>(N, N), range<2>(1, N));
        h.parallel_for(R, [=] (nd_item<2> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t r = item.get_global_id()[0];
            size_t c = item.get_global_id()[1];
            
            if (MATR[r][c]==0 && ROW_C[r]==0 && COL_C[c]==0) {
                if (r < ROW[0]) {
                    ROW[0] = int(r);
                    COL[0] = int(c);
                }
                if (r == ROW[0] && c < COL[0]) {
                    COL[0] = int(c);
                }
            }
        });
    }).wait();
}

//this function also updates col with the position of the star, if found
bool there_is_star_in_row(buffer<int, 1> &buf_ROW, buffer<int, 1> buf_COL, buffer<int, 2> &buf_MASK, queue &Q) {

    host_accessor ROW(buf_ROW, read_only);
    buffer buf_MASK_ROW{buf_MASK, id{size_t(ROW[0]), 0}, range{1, N}};
    
    buffer<bool, 1> buf_RESULT{range<1>(1)};
    
    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h);
        
        accessor MASK_ROW (buf_MASK_ROW, h, read_only);
        accessor COL (buf_COL, h, write_only);
        accessor RESULT (buf_RESULT, h, write_only, no_init);
        
        auto R = nd_range<1>(range<1>(N), range<1>(N));
        h.parallel_for(R, [=] (nd_item<1> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            
            size_t idx = item.get_local_id()[0];
            auto grp = item.get_group();
            
            //it is guaranteed that there is at most 1 star per row
            if (MASK_ROW[0][idx] == 1) {
                COL[0] = idx;
                RESULT[0] = true;
            }
            if (none_of_group(grp, MASK_ROW[0][idx]==1)){
                RESULT[0] = false;
            }
        });
    }).wait();
    
    host_accessor result(buf_RESULT, read_only);
    
    return result[0];
}



void step_towards_optimality(buffer<int, 2> &buf_MATRIX, buffer<bool, 1> &buf_ROW_COVER, buffer<bool, 1> &buf_COL_COVER, queue &Q) {
    
    int n_row_covered[1];
    int n_col_covered[1];

    buffer<int, 1> buf_N_ROW{&n_row_covered[0], range<1>(1), props};
    buffer<int, 1> buf_N_COL{&n_col_covered[0], range<1>(1), props};
    
    host_accessor N_ROW(buf_N_ROW, read_only);
    host_accessor N_COL(buf_N_COL, read_only);
    
    buffer<int, 1> buf_DELTA_ROW{range<1>(N-N_ROW[0]), props};
    buffer<int, 1> buf_DELTA_COL{range<1>(N-N_COL[0]), props};

    buffer<int, 1> buf_MINIMUM{range<1>(1), props};
    
    //this kernel finds the n° of covered rows and columns
    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h);
        
        accessor ROW (buf_ROW_COVER, h, read_only);
        accessor COL (buf_COL_COVER, h, read_only);
        accessor N_ROW (buf_N_ROW, h, write_only, no_init);
        accessor N_COL (buf_N_COL, h, write_only, no_init);
        
        auto R = nd_range<1>(range<1>(N), range<1>(N));
        h.parallel_for(R, [=] (nd_item<1> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t idx = item.get_local_id()[0];
            auto grp = item.get_group();
            
            N_ROW[0] = reduce_over_group(grp, int(ROW[idx]), sycl::plus<>());
            N_COL[0] = reduce_over_group(grp, int(COL[idx]), sycl::plus<>());
        });
    }).wait();

    
    //TODO: check whether it's possible to avoid building the deltas, and instead just masking the matrix cells from the reduce.minimum by using if statements
    //this kernel build the shifting step (DELTA_ROW/DELTA_COL) to be used in the next kernel,
    //to pick only the uncovered cells on which to compute the minimum
    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h); //TODO: remove
        
        accessor ROW (buf_ROW_COVER, h, read_only);
        accessor COL (buf_COL_COVER, h, read_only);
        accessor DELTA_ROW (buf_DELTA_ROW, h, write_only, no_init);
        accessor DELTA_COL (buf_DELTA_COL, h, write_only, no_init);
        
        auto R = nd_range<1>(range<1>(N), range<1>(N));
        h.parallel_for(R, [=] (nd_item<1> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t idx = item.get_local_id()[0];
            auto grp = item.get_group();
            
            
            int delta_row = inclusive_scan_over_group (grp, int(ROW[idx]), sycl::plus<>());
            if (ROW[idx]==0) {
                DELTA_ROW[idx-delta_row] = delta_row;
            }
            
            int delta_col = inclusive_scan_over_group (grp, int(COL[idx]), sycl::plus<>());
            if (COL[idx]==0) {
                DELTA_COL[idx-delta_col] = delta_col;
            }
        });
    }).wait();
    
    
    //this kernel finds the minimum over the uncovered sub-matrix
    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h);
        
        accessor MATRIX (buf_MATRIX, h, read_only);
        accessor DELTA_ROW (buf_DELTA_ROW, h, read_only);
        accessor DELTA_COL (buf_DELTA_COL, h, read_only);
        accessor MINIMUM (buf_MINIMUM, h, write_only, no_init);

        // N-N_ROW[0] has to be < than work_group_size
        auto R = nd_range<2>(range<2>(N-N_ROW[0], N-N_COL[0]), range<2>(N-N_ROW[0], N-N_COL[0]));
        h.parallel_for(R, [=] (nd_item<2> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t r = item.get_global_id()[0];
            size_t c = item.get_global_id()[1];
                    
            auto grp = item.get_group();
            group_barrier(grp);
            
            MINIMUM[0] = reduce_over_group (grp, MATRIX[r+DELTA_ROW[r]][c+DELTA_COL[c]], sycl::minimum<>());
            
        });
    }).wait();

    
    //This kernel adds the minimum to twice-covered cells, and subtracts it from uncovered ones
    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h); //TODO: remove
        
        accessor MATRIX (buf_MATRIX, h, write_only);
        accessor ROW (buf_ROW_COVER, h, read_only);
        accessor COL (buf_COL_COVER, h, read_only);
        accessor MINIMUM (buf_MINIMUM, h, read_only);
        
        auto R = nd_range<2>(range<2>(N, N), range<2>(1, N));
        h.parallel_for(R, [=] (nd_item<2> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t r = item.get_global_id()[0];
            size_t c = item.get_global_id()[1];
            
            if (ROW[r]==0 && COL[c]==0) {
                MATRIX[r][c] -= MINIMUM[0];
            }
            if (ROW[r]==1 && COL[c]==1) {
                MATRIX[r][c] += MINIMUM[0];
            }
        });
    }).wait();
}



//Prototypes to be used in alternating_path
void find_star_in_col (int col, buffer<int, 2> &buf_MASK, buffer<int, 1> &buf_ROW_IDX, queue &Q);
void find_prime_in_row (int row, buffer<int, 2> &buf_MASK, buffer<int, 1> &buf_COL_IDX, queue &Q);
void augment_path (buffer<int, 2> &buf_PATH, buffer<int, 2> &buf_MASK, queue &Q);

void alternating_path(buffer<int, 2> &buf_PATH, buffer<int, 1> &buf_P_ROW, buffer<int, 1> &buf_P_COL, buffer<int, 2> &buf_MASK, queue &Q) {

    int row[1] = {-1};
    int col[1] = {-1};
    int path_count = 1;
    bool done = false;

    buffer<int, 1> buf_ROW_IDX{&row[0], range<1>(1), props};
    buffer<int, 1> buf_COL_IDX{&col[0], range<1>(1), props};

    host_accessor ROW_IDX(buf_ROW_IDX, read_only);
    host_accessor COL_IDX(buf_COL_IDX, read_only);
    
    host_accessor PATH(buf_PATH, read_write);

    
    //reset path to 0, except the first row, which contains path_row_0 and path_col_0 
    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h);

        accessor PATH (buf_PATH, h, write_only, no_init);
        accessor P_ROW (buf_P_ROW, h, read_only);
        accessor P_COL (buf_P_COL, h, read_only);

        auto R = nd_range<2>(range<2>(2*N, 2), range<2>(N, 1));
        h.parallel_for(R, [=] (nd_item<2> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t r = item.get_global_id()[0];
            size_t c = item.get_global_id()[1];

            if (r!=0) {
                PATH[r][c] = 0;
            }
            if (r==0) {
                if (c==0) {
                    PATH[r][c] = P_ROW[0];
                }
                if (c==1) {
                    PATH[r][c] = P_COL[0];
                }
            }
        });
    }).wait();


    while(!done) {
        find_star_in_col(PATH[path_count - 1][1], buf_MASK, buf_ROW_IDX, Q);
        //cout << "Found a star in [" << row[0] << "," << PATH[path_count - 1][1] << "]\n";
        if (ROW_IDX[0] > -1) {
            path_count += 1;
            PATH[path_count - 1][0] = ROW_IDX[0];
            PATH[path_count - 1][1] = PATH[path_count - 2][1];
        }
        else {
            done = true;
        }

        if (!done) {
            find_prime_in_row(PATH[path_count - 1][0], buf_MASK, buf_COL_IDX, Q);
            //cout << "Found a prime in [" << PATH[path_count - 1][0] << "," << col[0] << "]\n";
            path_count += 1;
            PATH[path_count - 1][0] = PATH[path_count - 2][0];
            PATH[path_count - 1][1] = COL_IDX[0];
        }
    }

    augment_path(buf_PATH, buf_MASK, Q);

    //erase all primes from the mask
    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h); //TODO: remove

        accessor MASK (buf_MASK, h, read_write);

        auto R = nd_range<2>(range<2>(N, N), range<2>(1, N));
        h.parallel_for(R, [=] (nd_item<2> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {

            size_t r = item.get_global_id()[0];
            size_t c = item.get_global_id()[1];

            if (MASK[r][c]==2) {
                MASK[r][c]=0;
            }
        });
    }).wait();
}

void find_star_in_col(int col, buffer<int, 2> &buf_MASK, buffer<int, 1> &buf_ROW_IDX, queue &Q) {

    int temp[1] = {col};
    buffer<int, 1> buf_COL{&temp[0], range<1>(1)};

    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h);
        
        accessor MASK (buf_MASK, h, read_only);
        accessor ROW (buf_ROW_IDX, h, write_only, no_init);
        accessor COL (buf_COL, h, read_only);

        auto R = nd_range<1>(range<1>(N), range<1>(N));
        h.parallel_for(R, [=] (nd_item<1> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t idx = item.get_local_id()[0];
            auto grp = item.get_group();

            bool no_stars = none_of_group(grp, MASK[idx][col] == 1);

            group_barrier(grp);

            if (no_stars) {
                if (idx==0) {
                    ROW[0] = -1;
                }
            }

            if (MASK[idx][col]==1) {
                ROW[0] = idx;
            }
        });
    }).wait();
}

void find_prime_in_row (int row, buffer<int, 2> &buf_MASK, buffer<int, 1> &buf_COL_IDX, queue &Q) {

    buffer buf_MASK_ROW{buf_MASK, id{size_t(row), 0}, range{1,N}};

    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h);
        
        accessor MASK_ROW (buf_MASK_ROW, h, read_only);
        accessor COL (buf_COL_IDX, h, write_only, no_init);

        auto R = nd_range<1>(range<1>(N), range<1>(N));
        h.parallel_for(R, [=] (nd_item<1> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t idx = item.get_local_id()[0];

            if (MASK_ROW[0][idx]==2) {
                COL[0] = idx;
            }
        });
    }).wait();
}

void augment_path (buffer<int, 2> &buf_PATH, buffer<int, 2> &buf_MASK, queue &Q) {

    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h);
        
        accessor PATH (buf_PATH, h, read_only);
        accessor MASK (buf_MASK, h, write_only);

        auto R = nd_range<1>(range<1>(2*N), range<1>(N));
        h.parallel_for(R, [=] (nd_item<1> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t path_row = item.get_global_id()[0];

            //the left column of PATH contains the row index of the mask
            //the right column of PATH contains the column index of the mask

            //if mask cointains a star, it is removed
            if (MASK[PATH[path_row][0]][PATH[path_row][1]] == 1) {
                MASK[PATH[path_row][0]][PATH[path_row][1]] = 0;
            }
            //if mask contains a prime, it becomes a star
            if (MASK[PATH[path_row][0]][PATH[path_row][1]] == 2) {
                MASK[PATH[path_row][0]][PATH[path_row][1]] = 1;
            }
        });
    }).wait();
}



void optimal_assignment(buffer<int, 2> buf_MATRIX, buffer<int, 2> buf_MASK, buffer<int, 2> buf_ASSIGNMENT, buffer<int, 1> buf_COST, queue &Q) {
    
    Q.submit([&] (handler &h) {
        //auto out = stream(12500, 768, h);

        accessor MASK (buf_MASK, h, read_only);
        accessor MATRIX (buf_MATRIX, h, read_only);
        accessor ASSIGNMENT (buf_ASSIGNMENT, h, write_only, no_init);
        accessor COST (buf_COST, h, write_only);

        auto R = nd_range<2>(range<2>(N, N), range<2>(1, N));
        h.parallel_for(R, [=] (nd_item<2> item) /*[[intel::reqd_sub_group_size(SUB_GROUP_SIZE_16)]]*/ {
            size_t r = item.get_global_id()[0];
            size_t c = item.get_global_id()[1];
            auto sg = item.get_sub_group();

            if (MASK[r][c] == 1) {
                ASSIGNMENT[r][0] = r;
                ASSIGNMENT[r][1] = c;

                //TODO: check if it works
                COST[0] += MATRIX[r][c];
            }
        });
    }).wait();
}










int main () {
    ///*** initialize the queue for parallelization, and display critical informations on the acceleration device ***///
    queue Q(cpu_selector_v, property::queue::in_order());

    //start measuring execution time
    auto start = std::chrono::steady_clock::now();
    
    cout << "Device: " << Q.get_device().get_info<info::device::name>() << "\n";
    
    //# get all supported sub_group sizes and print
    auto sg_sizes = Q.get_device().get_info<info::device::sub_group_sizes>();
    std::cout << "Supported Sub-Group Sizes : ";
    for (int i=0; i<sg_sizes.size(); i++) std::cout << sg_sizes[i] << " "; std::cout << "\n";
    
    //# find out maximum supported sub_group size
    auto max_sg_size = std::max_element(sg_sizes.begin(), sg_sizes.end());
    std::cout << "Max Sub-Group Size        : " << max_sg_size[0] << "\n";

    cout << "Device max work group size: " << Q.get_device().get_info<info::device::max_work_group_size>() << "\n";
    cout << "Device max work item dimensions: " << Q.get_device().get_info<info::device::max_work_item_dimensions>() << "\n";
    

    
    ///*** declarations and initializations ***///
    
    int original_matrix[N][N] = {
        {1, 2, 3, 4, 5},
        {1, 2, 2, 2, 2},
        {4, 6, 8, 10, 12},
        {0, 3, 5, 7, 9},
        {10, 9, 8, 7, 6}
    };

    int matrix[N][N];

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            matrix[i][j] = original_matrix[i][j];
        }
    }

    
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            cout << matrix[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";
    
    int mask[N][N];
    bool row_cover[N];
    bool col_cover[N];
    
    bool found[1] = {false};
    bool optimize = false;
    int path_row_0[1];
    int path_col_0[1];
    int path[2*N][2]; //each row is the tuple of coordinates that build the alternating path
    int assignment[N][2];
    int cost[1] = {0};
    

    {
        ///*** buffers declarations and initializations ***///    
        buffer<int, 2> buf_MATRIX{&matrix[0][0], range<2>(N, N), props};
        buffer<int, 2> buf_MASK{&mask[0][0], range<2>(N, N), props};
        buffer<bool, 1> buf_ROW_COVER{&row_cover[0], range<1>(N), props};
        buffer<bool, 1> buf_COL_COVER{&col_cover[0], range<1>(N), props};
        buffer<bool, 1> buf_FOUND{&found[0], range<1>(1), props};

        host_accessor FOUND(buf_FOUND, read_only);
        
        /*** pre-processing of the matrix ***/
        row_reduction (buf_MATRIX, Q);
        column_reduction (buf_MATRIX, Q);
        
    
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){
                cout << matrix[i][j] << " ";
            }
            cout << "\n";
        }
        cout << "\n";
        
    
        
        cout << "Starting Hungarian Algorithm..." << std::endl;
    
        //cout << "Starring the zeroes..." << std::endl;
        starring_the_zeroes(buf_MATRIX, buf_MASK, buf_COL_COVER, Q);
        cout << "First starring of the zeroes successful\n";
        
        while (!FOUND[0]) {
            all_columns_covered(buf_MASK, buf_ROW_COVER, buf_COL_COVER, buf_FOUND, Q);
            cout << "All columns are covered: " << (found[0]?"T":"F") << std::endl;

            
            
            if (FOUND[0]){
                break;
            }
            else {
                buffer<int, 1> buf_PATH_ROW_0{&path_row_0[0], range<1>(1), props};
                buffer<int, 1> buf_PATH_COL_0{&path_col_0[0], range<1>(1), props};
                
                find_prime_and_uncover_star(buf_MATRIX, buf_MASK, buf_ROW_COVER, buf_COL_COVER, buf_PATH_ROW_0, buf_PATH_COL_0, optimize, Q);
                cout << "Start optimization step: " << (optimize?"Y":"N") << std::endl;
                
                if (optimize) {
                    step_towards_optimality(buf_MATRIX, buf_ROW_COVER, buf_COL_COVER, Q);
                    starring_the_zeroes(buf_MATRIX, buf_MASK, buf_COL_COVER, Q);
                }
                else {
                    buffer<int, 2> buf_PATH{&path[0][0], range<2>(2*N, 2), props};
                    
                    alternating_path(buf_PATH, buf_PATH_ROW_0, buf_PATH_COL_0, buf_MASK, Q);
                }
            }
        }
        buffer<int, 2> buf_ORIGINAL_MATRIX{&original_matrix[0][0], range<2>(N, N), props};
        buffer<int, 2> buf_ASSIGNMENT{&assignment[0][0], range<2>(N, 2), props};
        buffer<int, 1> buf_COST{&cost[0], range<1>(1), props};
        
        optimal_assignment(buf_ORIGINAL_MATRIX, buf_MASK, buf_ASSIGNMENT, buf_COST, Q);
    }


            
    //take the time at which the algorythm has ended
    auto end = std::chrono::steady_clock::now();

    cout << "Optimal assignment: \n";
    for (int i = 0; i < N; i++) {
        cout << "(" << assignment[i][0] << ", " << assignment[i][1] << ") \n";
    } cout << std::endl;

    cout << "Total cost: " << cost[0] << std::endl;

    cout << "Execution time: " << (end-start).count()/1e6 << " microseconds" << std::endl;
    
    return 0;
}
