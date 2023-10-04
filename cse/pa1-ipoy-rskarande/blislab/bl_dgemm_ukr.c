#include "bl_config.h"
#include "bl_dgemm_kernel.h"
#include <arm_sve.h>
#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based micorkernel
//
void bl_dgemm_ukr( int    k,
		   int    m,
                   int    n,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        for ( j = 0; j < n; ++j )
        { 
            for ( i = 0; i < m; ++i )
            { 
	      // ldc is used here because a[] and b[] are not packed by the
	      // starter code
	      // cse260 - you can modify the leading indice to DGEMM_NR and DGEMM_MR as appropriate
	      //
	      c( i, j, ldc ) += a( i, l, ldc) * b( l, j, ldc );   
            }
        }
    }

}


// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//

// slow kernel for packing implementation
void pack_ukr( int    k,
		   int    m,
                   int    n,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        for ( j = 0; j < n; ++j )
        { 
            for ( i = 0; i < m; ++i )
            { 
	      // ldc is used here because a[] and b[] are not packed by the
	      // starter code
	      // cse260 - you can modify the leading indice to DGEMM_NR and DGEMM_MR as appropriate
	      //
	      c( i, j, ldc ) += a[ (l)*(DGEMM_MR) + (i) ] * b[ (l)*(DGEMM_NR) + (j) ];
          
            }
        }
    }

}

void sve_ukr( int    k,
		   int    m,
                   int    n,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l;

    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x;
    svbool_t pred = svwhilelt_b64_u64(0, 4);

    c0x = svld1_f64(pred, c);
    c1x = svld1_f64(pred, c + ldc);
    c2x = svld1_f64(pred, c + ldc*2);
    c3x = svld1_f64(pred, c + ldc*3);
    
    for ( l = 0; l < k; l++ )
    {    
        register float64_t aval = a[l*DGEMM_MR];
        ax = svdup_f64(aval);
        bx = svld1_f64(svptrue_b64(), &b[l*DGEMM_NR]);
  
        c0x = svmla_f64_m(pred, c0x, bx, ax); // row 1

        aval = a[1 + DGEMM_MR *l];
        ax =svdup_f64(aval);
        c1x = svmla_f64_m(pred, c1x, bx, ax); // row 2
   
        aval = a[2 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c2x = svmla_f64_m(pred, c2x, bx, ax); // row 3
        
        aval = a[3 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c3x = svmla_f64_m(pred, c3x, bx, ax); // row 4
       
    }

    svst1_f64(pred, c, c0x);
    svst1_f64(pred, c + ldc, c1x);
    svst1_f64(pred, c + 2*ldc, c2x);
    svst1_f64(pred, c + 3*ldc, c3x);
}

void sve_ukr2( int    k,
		   int    m,
                   int    n,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l;

    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x;
    svbool_t pred = svwhilelt_b64_u64(0, 4);

    c0x = svld1_f64(pred, c);
    c1x = svld1_f64(pred, c + ldc);
    c2x = svld1_f64(pred, c + ldc*2);
    c3x = svld1_f64(pred, c + ldc*3);
    c4x = svld1_f64(pred, c + ldc*4);
    c5x = svld1_f64(pred, c + ldc*5);
    c6x = svld1_f64(pred, c + ldc*6);
    c7x = svld1_f64(pred, c + ldc*7);
    
    for ( l = 0; l < k; l++ )
    {    
        register float64_t aval = a[l*DGEMM_MR];
        ax = svdup_f64(aval);
        bx = svld1_f64(svptrue_b64(), &b[l*DGEMM_NR]);
  
        c0x = svmla_f64_m(pred, c0x, bx, ax); // row 1

        aval = a[1 + DGEMM_MR *l];
        ax =svdup_f64(aval);
        c1x = svmla_f64_m(pred, c1x, bx, ax); // row 2
   
        aval = a[2 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c2x = svmla_f64_m(pred, c2x, bx, ax); // row 3
        
        aval = a[3 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c3x = svmla_f64_m(pred, c3x, bx, ax); // row 4

        aval = a[4 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c4x = svmla_f64_m(pred, c4x, bx, ax); // row 5
        
        aval = a[5 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c5x = svmla_f64_m(pred, c5x, bx, ax); // row 6

        aval = a[6 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c6x = svmla_f64_m(pred, c6x, bx, ax); // row 7
        
        aval = a[7 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c7x = svmla_f64_m(pred, c7x, bx, ax); // row 8
       
    }

    svst1_f64(pred, c, c0x);
    svst1_f64(pred, c + ldc, c1x);
    svst1_f64(pred, c + 2*ldc, c2x);
    svst1_f64(pred, c + 3*ldc, c3x);
    svst1_f64(pred, c + 4*ldc, c4x);
    svst1_f64(pred, c + 5*ldc, c5x);
    svst1_f64(pred, c + 6*ldc, c6x);
    svst1_f64(pred, c + 7*ldc, c7x);
}

void sve_ukr3( int    k,
		   int    m,
                   int    n,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l;

    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x, c8x, c9x, c10x, c11x, c12x, c13x, c14x, c15x;
    svbool_t pred = svwhilelt_b64_u64(0, 4);

    c0x = svld1_f64(pred, c);
    c1x = svld1_f64(pred, c + ldc);
    c2x = svld1_f64(pred, c + ldc*2);
    c3x = svld1_f64(pred, c + ldc*3);
    c4x = svld1_f64(pred, c + ldc*4);
    c5x = svld1_f64(pred, c + ldc*5);
    c6x = svld1_f64(pred, c + ldc*6);
    c7x = svld1_f64(pred, c + ldc*7);

    c8x = svld1_f64(pred, c + ldc*8);
    c9x = svld1_f64(pred, c + ldc*9);
    c10x = svld1_f64(pred, c + ldc*10);
    c11x = svld1_f64(pred, c + ldc*11);
    // c12x = svld1_f64(pred, c + ldc*12);
    // c13x = svld1_f64(pred, c + ldc*13);
    // c14x = svld1_f64(pred, c + ldc*14);
    // c15x = svld1_f64(pred, c + ldc*15);
    
    for ( l = 0; l < k; l++ )
    {    
        register float64_t aval = a[l*DGEMM_MR];
        ax = svdup_f64(aval);
        bx = svld1_f64(svptrue_b64(), &b[l*DGEMM_NR]);
  
        c0x = svmla_f64_m(pred, c0x, bx, ax); // row 1

        aval = a[1 + DGEMM_MR *l];
        ax =svdup_f64(aval);
        c1x = svmla_f64_m(pred, c1x, bx, ax); // row 2
   
        aval = a[2 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c2x = svmla_f64_m(pred, c2x, bx, ax); // row 3
        
        aval = a[3 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c3x = svmla_f64_m(pred, c3x, bx, ax); // row 4

        aval = a[4 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c4x = svmla_f64_m(pred, c4x, bx, ax); // row 5
        
        aval = a[5 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c5x = svmla_f64_m(pred, c5x, bx, ax); // row 6

        aval = a[6 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c6x = svmla_f64_m(pred, c6x, bx, ax); // row 7
        
        aval = a[7 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c7x = svmla_f64_m(pred, c7x, bx, ax); // row 8


        aval = a[8 + DGEMM_MR *l];
        ax =svdup_f64(aval);
        c8x = svmla_f64_m(pred, c8x, bx, ax); // row 9
   
        aval = a[9 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c9x = svmla_f64_m(pred, c9x, bx, ax); // row 10
        
        aval = a[10 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c10x = svmla_f64_m(pred, c10x, bx, ax); // row 11

        aval = a[11 + DGEMM_MR*l];
        ax =svdup_f64(aval);
        c11x = svmla_f64_m(pred, c11x, bx, ax); // row 12
        
        // aval = a[12 + DGEMM_MR*l];
        // ax =svdup_f64(aval);
        // c12x = svmla_f64_m(pred, c12x, bx, ax); // row 13

        // aval = a[13 + DGEMM_MR*l];
        // ax =svdup_f64(aval);
        // c13x = svmla_f64_m(pred, c13x, bx, ax); // row 14
        
        // aval = a[14 + DGEMM_MR*l];
        // ax =svdup_f64(aval);
        // c14x = svmla_f64_m(pred, c14x, bx, ax); // row 15

        // aval = a[15 + DGEMM_MR*l];
        // ax =svdup_f64(aval);
        // c15x = svmla_f64_m(pred, c15x, bx, ax); // row 16
       
    }

    svst1_f64(pred, c, c0x);
    svst1_f64(pred, c + ldc, c1x);
    svst1_f64(pred, c + 2*ldc, c2x);
    svst1_f64(pred, c + 3*ldc, c3x);
    svst1_f64(pred, c + 4*ldc, c4x);
    svst1_f64(pred, c + 5*ldc, c5x);
    svst1_f64(pred, c + 6*ldc, c6x);
    svst1_f64(pred, c + 7*ldc, c7x);

    svst1_f64(pred, c + 8*ldc, c8x);
    svst1_f64(pred, c + 9*ldc, c9x);
    svst1_f64(pred, c + 10*ldc, c10x);
    svst1_f64(pred, c + 11*ldc, c11x);
    // svst1_f64(pred, c + 12*ldc, c12x);
    // svst1_f64(pred, c + 13*ldc, c13x);
    // svst1_f64(pred, c + 14*ldc, c14x);
    // svst1_f64(pred, c + 15*ldc, c15x);
}

