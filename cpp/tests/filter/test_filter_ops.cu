/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
 *     Copyright 2018 Felipe Aramburu <felipe@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "helper/utils.cuh"

#include "test_filter_ops.cuh"

#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/valid_vectors.h>

#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <random>

using cudf::test::column_wrapper;

// Note: A few fixed lengths are not sufficient to test gdf_apply_boolean_mask.
// Multiple lengths are necessary to cover the large number of cases
struct lengths {

    enum : gdf_size_type {
        short_non_round = 25,
        short_round = 32,
        long_round = 100000, // Not currently used
        long_non_round = 100007,
    };
};

enum : gdf_bool { gdf_false = 0, gdf_true = 1 };

/*
const auto random<bool> = [] (size_t idx, size_t length, gdf_valid_type* const valids) {
    if (std::rand()/(float)(RAND_MAX + 1u) < 0.5) {
        gdf::util::turn_bit_on(valids, idx);
    }
    else {
        gdf::util::turn_bit_off(valids, idx);
    }
};
*/

template <typename T>
struct random {
    static std::random_device device {};
    static std::mt19937 generator { device() };
    static std::uniform_int_distribution<T> distribution {};
    // TODO: Maybe I need a constructor?
    bool random(gdf_size_type) noexcept(noexcept(distribution(generator)))
    {
        return distribution(generator);
    }
};

template <typename T, T Value>
T constant(gdf_size_type) noexcept { return Value; }

using fully_valid = fully_valid;


struct first_half {
    gdf_size_type range_length;
    bool operator()(gdf_size_type i) const noexcept {
        return (i < range_length / 2);
    }
};

template <gdf_size_type CycleLength>
struct first_of_every_k {
    T operator()(gdf_size_type i) const noexcept {
        return i % CycleLength;
    }
};


template <typename DataColumnElement>
struct ApplyBooleanMaskTest : public GdfTest {

    using element_type = DataColumnElement;

    // contains the input column
    std::vector<DataColumnElement> host_vector;

    // contains valids for column
    host_valid_pointer host_valid = nullptr;

    // contains the stencil column
    std::vector<gdf_bool> host_stencil_vector;

    // contains valids for stencil column
    host_valid_pointer host_stencil_valid = nullptr;

    gdf_col_pointer col;
    gdf_col_pointer stencil;
    gdf_col_pointer output;

    // contains the filtered column
    std::vector<DataColumnElement> reference_vector;

    // contains the valids for the filtered column
    host_valid_pointer reference_valid = nullptr;

    gdf_size_type reference_null_count = 0;

    DataColumnElement maximum_value = 100;


    ApplyBooleanMaskTest()
    {
        // Use constant seed so the psuedo-random order is the same each time
        // Each time the class is constructed a new constant seed is used
        static size_t number_of_instantiations{0};
        std::srand(number_of_instantiations++);
    }

    ~ApplyBooleanMaskTest()
    {
    }
    /* --------------------------------------------------------------------------*/
    /**
    * @Synopsis  Initializes input columns
    *
    * @Param length The length of the column
    * @Param valids_col The type of initialization for valids from col according the valids_t enum
    * @Param valids_stencil The type of initialization for valids from stencil according the valids_t enum
    * @Param print Optionally print column for debugging
    */
    /* ----------------------------------------------------------------------------*/
    void create_input(
        const size_t                 length,
        valids_functor_initializer   valids_col,
        stencil_functor_initializer  stencil_vals,
        valids_functor_initializer   valids_stencil,
        bool                         print = false)
    {

        initialize_values(host_vector, length, maximum_value);
        if (valids_col) {
            initialize_valids(host_valid, length, valids_col);
        }

        col = create_gdf_column(host_vector, host_valid);

        initialize_stencil_values(host_stencil_vector, length, stencil_vals);
        if (valids_stencil) {
            initialize_valids(host_stencil_valid, length, valids_stencil);
        }
        std::cout << std::endl;

        stencil = create_gdf_column(host_stencil_vector, host_stencil_valid);
        // TODO: Add ability to set arbitrary or random inputs as well

        std::vector<DataColumnElement> zero_vector(length, 0);
        host_valid_pointer output_valid;

        initialize_valids(output_valid, length, fully_valid);

        output = create_gdf_column(zero_vector, output_valid);

        allocate_reference_valids(reference_valid, length);

        if(print) {
            std::cout<<"Input (" << length << " elements):\n";
            print_gdf_column(col.get(), this->element_print_width());

            std::cout<<"Stencil (" << length << " bit-packed elements):\n";
            print_gdf_column(stencil.get(), this->element_print_width());

            std::cout<<"\n";
        }
    }

    unsigned element_print_width() const {
        return std::ceil(std::log10(maximum_value));
    }

    void print_reference_column() {
        std::cout<<"Reference Output:\n";
        print_typed_column<DataColumnElement>(reference_vector.data(), reference_valid.get(), reference_vector.size(), element_print_width());
//        for(size_t i = 0; i < reference_vector.size(); i++) {
//            std::cout << "(" << std::to_string(reference_vector[i]) << "|" << gdf_is_valid(reference_valid.get(), i) << "), ";
//        }
        std::cout<<"\n\n";
    }

    void allocate_reference_valids(host_valid_pointer& valid_ptr, size_t length) {
        auto deleter = [](gdf_valid_type* valid) { delete[] valid; };
        auto n_bytes = get_number_of_bytes_for_valid(length);
        auto valid_bits = new gdf_valid_type[n_bytes];
    
        valid_ptr = host_valid_pointer{ valid_bits, deleter };
    }

    void compute_reference_solution() {

        size_t valid_index=0;
        for (size_t index = 0 ; index < host_vector.size() ; index++) {
//            std::cout << std::setw(2) << index << ": stencil = " << (int) host_stencil_vector[index] << " stencil validity = " << (int) gdf_is_valid(host_stencil_valid.get(), index)
//                << " data = " << (int) host_vector[index] << " data validity = " <<  (int) gdf_is_valid(host_valid.get(), index) << '\n';
            if (host_stencil_vector[index] != 0 &&
                gdf_is_valid(host_stencil_valid.get(), index) )
            {
                reference_vector.push_back(host_vector[index]);

                if ( gdf_is_valid(host_valid.get(), index) ) {
                    gdf::util::turn_bit_on(reference_valid.get(), valid_index);
                } else {
                    gdf::util::turn_bit_off(reference_valid.get(), valid_index);
                    reference_null_count++;
                }

                valid_index++;
            }
        }
    }

    gdf_error compute_gdf_result() {
        gdf_error error = gdf_apply_boolean_mask(col.get(), stencil.get(), output.get());
        return error;
    }

    void print_debug() {
        std::cout<<"Output:\n";
        print_gdf_column(output.get(), element_print_width());
        std::cout<<"\n";

        print_reference_column();
    }

    void compare_gdf_result() {
        std::vector<DataColumnElement> host_result(reference_vector.size());
        
        // Copy result of applying stencil to the host
        EXPECT_EQ(cudaMemcpy(host_result.data(), output.get()->data, output.get()->size * sizeof(DataColumnElement), cudaMemcpyDeviceToHost), cudaSuccess);

        // Compare the gpu and reference solutions
        for(size_t i = 0; i < reference_vector.size(); ++i) {
            EXPECT_EQ(reference_vector[i], host_result[i]);
        }

        auto n_bytes = get_number_of_bytes_for_valid(col->size);
        gdf_valid_type* host_ptr = new gdf_valid_type[n_bytes];

        // Copy valids to the host
        EXPECT_EQ(cudaMemcpy(host_ptr, output.get()->valid, n_bytes, cudaMemcpyDeviceToHost), cudaSuccess);

        // Compare the gpu and reference valid arrays
        for(size_t i = 0; i < reference_vector.size(); ++i) {
            EXPECT_EQ( gdf_is_valid(reference_valid.get(), i), gdf_is_valid(host_ptr, i) );
        }

        // Check null count
        EXPECT_EQ(output.get()->null_count, reference_null_count);
    }
};

typedef ::testing::Types<
    int32_t,
    int64_t,
    float,
    double
  > Implementations;

TYPED_TEST_CASE(ApplyBooleanMaskTest, Implementations);

//Todo: usage_example
//TYPED_TEST(ApplyBooleanMaskTest, usage_example) {


TYPED_TEST(ApplyBooleanMaskTest, all_multiple_32) {

    auto data    = column_wrapper<element_type>(lengths::short_non_round, random<element_type>,          fully_valid);
    auto stencil = column_wrapper<gdf_bool    >(lengths::short_non_round, constant<gdf_bool, gdf_true>,  fully_valid);
    auto output  = column_wrapper<element_type>(lengths::short_non_round, constant<element_type>,        fully_valid);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );

    this->compute_reference_solution();

    expect_columns_are_equal(output, reference_) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, all_non_multiple_of_32) {
    const bool print_result{false};

    auto data    = column_wrapper<element_type>(lengths::short_non_round, random_value_initializer, fully_valid);
    auto stencil = column_wrapper<gdf_bool    >(lengths::short_non_round, func_stencil_all,         fully_valid);
    auto output  = column_wrapper<element_type>(lengths::short_non_round, func_zero,                fully_valid);

    ASSERT_CUDF_SUCCESS(
       gdf_apply_boolean_mask(data.get(), stencil.get(), output.get())
    );

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, half_all_third_multiple_32) {
    const bool print_result{false};

    this->create_input(lengths::short_round, func_first_half_bits_on, func_stencil_all, func_every_third_bit_on, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, half_all_third_non_multiple_32) {
    const bool print_result{false};

    this->create_input(lengths::short_non_round, func_first_half_bits_on, func_stencil_all, func_every_third_bit_on, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, all_half_all_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(lengths::short_non_round, fully_valid, func_stencil_first_half, fully_valid, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, all_random_all_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(lengths::short_non_round, fully_valid, random<gdf_bool>, fully_valid, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, all_third_random_multiple_of_32) {
    const bool print_result{false};

    this->create_input(lengths::short_round, fully_valid, func_stencil_third, random<bool>, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, random_random_random_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(lengths::short_non_round, random<bool>, random<gdf_bool>, random<bool>, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, none_all_all_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(lengths::short_non_round, nullptr, func_stencil_all, fully_valid, print_result);

    gdf_error error = this->compute_gdf_result();
    if (error != GDF_SUCCESS) {
        std::cout << "gdf_apply_boolean_mask returned the error: " << gdf_error_get_name(error) << '\n';
    }
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, all_none_all_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(lengths::short_non_round, fully_valid, func_stencil_none, fully_valid, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, all_all_none_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(lengths::short_non_round, fully_valid, func_stencil_all, nullptr, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, none_none_none_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(lengths::short_non_round, nullptr, func_stencil_none, nullptr, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, all_random_random_big_input) {
    const bool print_result{false};

    this->create_input(lengths::long_non_round, fully_valid, random<gdf_bool>, random<bool>, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(ApplyBooleanMaskTest, all_random_random_empty_input) {
    const bool print_result{false};

    this->create_input(0, fully_valid, random<gdf_bool>, random<bool>, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ((size_t) this->output.get()->size, (size_t) this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}
