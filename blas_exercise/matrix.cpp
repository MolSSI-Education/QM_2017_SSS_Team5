// This code reads outputs from Python and do energy calculation.

#include <iostream>
#include <fstream>
#include <iomanip>
#include <lawrap/blas.h>
#include <vector>

using namespace std;

int main()
{
    using namespace std;
    int nbas = 7;
    int nocc = 5;

    ifstream file1("C.data");
    ifstream file2("S.data");
    ifstream file3("F.data");
    ifstream file4("H.data");


    // TODO : Use vector matrix instead array.
	double C[nbas][nbas];
	double Cocc[nocc][nbas]; // Notice the problem is
	double S[nbas][nbas];
	double F[nbas][nbas];
	double H[nbas][nbas];
	double temp;

	// Reads data
	for (int i = 0; i< nbas; ++i)
	{
		for (int j = 0;j < nbas; ++j)
		{
			file1 >> temp;
			file2 >> S[i][j];
			file3 >> F[i][j];
			file4 >> H[i][j];
			C[i][j] = temp;

			// Get Cocc for j<nocc
			if (j<nocc)
			{
				Cocc[j][i] = temp;
			}
		}
	}



		// Print Cocc matrix
		/*
    		std::cout<<"\n Check out the Cocc matrix: \n";
    		for (int i=0;i<nbas;++i)
    		{
    			for (int j=0;j<nocc;++j)
    			{
    				std::cout<<  std::setprecision(4) << std::fixed << Cocc[j][i] << "\t";
    			}
    			std::cout<< "\n";
    		}
		 */


// Calculate D = 2 * Cocc * Cocc.T
// It is been forced to be transformed to a double pointer.

    		double D[nbas][nbas];
    		LAWrap::gemm('N', 'T', nbas, nbas, nocc, 2.0, (double*) Cocc, nbas , (double*) Cocc, nbas, 0.0, *D, nbas);


		// Print D matrix
    		/*
		std::cout<<" \n Check out the D matrix: \n";

		    		for (int i=0;i< nbas;++i)
		    		{
		    			for (int j=0;j< nbas;++j)
		    			{
		    				std::cout<<  std::setprecision(4) << std::fixed << D[i][j] << "\t";
		    			}
		    			std::cout<< "\n";
		    		}
		*/

// Multiple D and S to get DS

    		double DS[nbas][nbas];
       	LAWrap::gemm('N', 'N', nbas, nbas, nbas, 1.0, (double*) D, nbas , (double*) S, nbas, 0.0, *DS, nbas);
    		double Tr = 0;
    		for (int i=0;i<nbas;++i)
    		{
    			Tr = Tr + DS[i][i];
    		}
    		std::cout<< "\n The trace of (D*S) is: " << Tr << "\n";


// Calculate SCF energy
    		// F+H
    		double FpH[nbas][nbas];
    		for (int i=0;i< nbas;++i)
    		{
    			for (int j=0;j< nbas;++j)
    			{
    				FpH[i][j] = F[i][j] + H[i][j];
    			}
    		}

    		// Print F+H
    		/*
    		std::cout<<" \n Check out the (F+H) matrix: \n";

    		    		for (int i=0;i< nbas;++i)
    		    		{
    		    			for (int j=0;j< nbas;++j)
    		    			{
    		    				std::cout<<  std::setprecision(4) << std::fixed << FpH[i][j] << "\t";
    		    			}
    		    			std::cout<< "\n";
    		    		}
		*/

    		// Calculate (F+H)*D
    		double FHD[nbas][nbas];

        LAWrap::gemm('N', 'N', nbas, nbas, nbas, 0.5, (double*) FpH, nbas , (double*) D, nbas, 0.0, *FHD, nbas);

		// Print (F+H)*D
        /*
		std::cout<<" \n Check out the (F+H)*D matrix: \n";

		    		for (int i=0;i< nbas;++i)
		    		{
		    			for (int j=0;j< nbas;++j)
		    			{
		    				std::cout<<  std::setprecision(4) << std::fixed << FHD[i][j] << "\t";
		    			}
		    			std::cout<< "\n";
		    		}
		*/

        	// Calculate Tr((FHD)) / 2
        double SCF = 0;
		for (int i=0;i< nbas;++i)
		{
				SCF = SCF + FHD[i][i] ;
		}

		double E_nuc = 8.002366450719078;
		SCF = SCF + E_nuc;
		std::cout<< "SCF energy is: " <<  std::setprecision(8) << std::fixed << SCF << " (Eh) \n";
}
