#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <string>
#include <iostream>


namespace py = pybind11;


long print_num(long s)
{
#ifdef _OPENMP
		std::cout << "Hello, OpenMP!" << std::endl;
		std::cout << "OpenMP will (max) use  " << omp_get_max_threads()<< " threads."<< std::endl;
#else
		std::cout << "Hello, World!"	 << std::endl;
		return 0;
#endif
		return 0;
std::cout << "Num argument: " << s << std::endl;
return s;
}



py::array_t<double> JK_numpy(int n,
                            py::array_t<double>p,
                            py::array_t<double>D,
                            int num )
{
    py::buffer_info p_info = p.request();
    py::buffer_info D_info = D.request();

    size_t J_nrows = p_info.shape[0];
    size_t J_ncols = p_info.shape[1];
    size_t nbf = n;

    const double * p_data = static_cast<double *>(p_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);

    std::vector<double>J_data(J_nrows*J_ncols);

    int m;
    if(num == 2)
    {
    for(size_t i = 0; i < J_nrows; i++)
    {
        for(size_t j = 0; j < J_ncols; j++)
        {
            double val = 0.0;
            for(size_t k = 0; k < nbf; k++)
            {
                for(size_t l = 0; l < nbf; l++)
                {
                        m = i * nbf * nbf *nbf + k * nbf * nbf +j * nbf +l;
                        val += p_data[m] * D_data[k*nbf+l];
                }
            }
            J_data[i*J_ncols+j] = val;

        }
     }
    }
    else if(num == 1)
    {
    for(size_t i = 0; i < J_nrows; i++)
    {
        for(size_t j = 0; j < J_ncols; j++)
        {
            double val = 0.0;
            for(size_t k = 0; k < nbf; k++)
            {
                for(size_t l = 0; l < nbf; l++)
                {
                    m = i * nbf * nbf *nbf + j * nbf * nbf +k * nbf +l;
                        val += p_data[m] * D_data[k*nbf+l];
                }
            }
            J_data[i*J_ncols+j] = val;

        }
     }
    }

    py::buffer_info Jbuf =
        {
            J_data.data(),
            sizeof(double),
            py::format_descriptor<double>::format(),
            2,
            { J_nrows, J_ncols },
            { J_nrows * sizeof(double), sizeof(double) }
        };

    return py::array_t<double>(Jbuf);
}



long factorial(long n)
{
	if(n < 0)
		throw std::runtime_error("n needs to be >= 0 for factorial");
	long fac = 1;
	for(long i = n; i > 0; i--)
		fac *= i;
	return fac;
}

long binomial_coefficient(long n, long k)
{	// (n,k) = n! /(k!(n-k)!)
	if(n < k)
		throw std::runtime_error("n needs to be >= k for factorial");
	long nk;
	nk = factorial(n)/(factorial(k)*factorial(n-k));
	return nk;
}

long double_factorial(long n)
{	// n!! = (n) * (n-2)*...
	// 0!! = 1
	// (-1)!! = 1
	if(n<-1)
	{
		throw std::runtime_error("n needs to be >= -1 for factorial");
	}

	long dfac = 1;
	for(long i = n; i > 0; i-=2)
		dfac *= i;
	return dfac;
}

double dot_product(const std::vector<double> &v1,
				   const std::vector<double> v2)
{
	if(v1.size()!= v2.size())
		throw std::runtime_error("Vectors are different lengths");

	if(v1.size() == 0)
		throw std::runtime_error("Zero-length vector");

	double dot = 0.0;
	for(size_t i=0; i<v1.size(); i++)
	{
		dot += v1[i] * v2[i];
	}
	return dot;
}

double dot_product_numpy(py::array_t<double> v1,
						 py::array_t<double> v2)
{
	py::buffer_info v1_info = v1.request();
	py::buffer_info v2_info = v2.request();

	if(v1_info.ndim !=1)
		throw std::runtime_error("v1 is not a vector");
	if(v2_info.ndim !=1)
		throw std::runtime_error("v2 is not a vector");
	if(v1_info.shape[0]!= v2_info.shape[0])
		throw std::runtime_error("v1 and v2 are not the same length");

	double dot = 0.0;

	const double * v1_data = static_cast<double *>(v1_info.ptr);
	const double * v2_data = static_cast<double *>(v2_info.ptr);

	for(size_t i = 0; i < v1_info.shape[0]; i++)
	{
		dot += v1_data[i] * v2_data[i];
		std::cout << v1_data[i]<<" and " << v2_data[i]<<  "\n";
	}

	return dot;
}

py::array_t<double> dgemm_numpy(double alpha,
								py::array_t<double> A,
								py::array_t<double> B)
		{

			py::buffer_info A_info = A.request();
			py::buffer_info B_info = B.request();

			if(A_info.ndim !=2)
				throw std::runtime_error("A is not a matrix");
			if(B_info.ndim !=2)
				throw std::runtime_error("B is not a matrix");
			if(A_info.shape[1]!= B_info.shape[0])
				throw std::runtime_error("Rows of A != Columns of B");

			size_t  C_nrows = A_info.shape[0];
			size_t  C_ncols = B_info.shape[1];
			size_t  n_k = A_info.shape[1]; //same as B_info.shape[0]


			const double * A_data = static_cast<double *>(A_info.ptr);
			const double * B_data = static_cast<double *>(B_info.ptr);

			std::vector<double> C_data(C_nrows * C_ncols);

			for(size_t i = 0; i< C_nrows; i++)
			{
				for(size_t j = 0; j < C_ncols; j++)
				{
					double val = 0.0;
					for(size_t k = 0; k < n_k; k++)
					{
						val += alpha * A_data[i*n_k + k] * B_data[k * C_ncols + j];
					}
					C_data[i*C_ncols + j] = val;
				}
			}

			py::buffer_info Cbuf =
			{
					C_data.data(),
					sizeof(double),
					py::format_descriptor<double>::format(),
					2,
					{ C_nrows, C_ncols },
					{ C_ncols * sizeof(double), sizeof(double) }
			};
			return py::array_t<double>(Cbuf);
		}


// The module to convert C++ into Python module
PYBIND11_PLUGIN(basic_mod)
{
	py::module m("basic_mod", "QM5's basic module");

	m.def("print_num", &print_num, "Prints the passed arg");
	m.def("factorial", &factorial, "Computes n!");
	m.def("binomial_coefficient", &binomial_coefficient, "Computes (n,k) = n! /(k!(n-k)!)");
	m.def("double_factorial", &double_factorial, "Computes n!!");
	m.def("dot_product", &dot_product);
	m.def("dot_product_numpy", &dot_product_numpy);
	m.def("dgemm_numpy", &dgemm_numpy);
	m.def("JK_numpy", &JK_numpy);
	
	return m.ptr();
}
