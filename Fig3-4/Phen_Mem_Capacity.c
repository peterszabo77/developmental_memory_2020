#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define AA 471
#define B 1586
#define CC 6988
#define DD 9689
#define M 16383
#define RIMAX 2147483648.0        /* = 2^31 */
#define RandomInteger (++nd, ra[nd & M] = ra[(nd-AA) & M] ^ ra[(nd-B) & M] ^ ra[(nd-CC) & M] ^ ra[(nd-DD) & M])
void seed(long seed);
static long ra[M+1], nd;

void seed(long seed)
{
 int  i;

 if(seed<0) { puts("SEED error."); exit(1); }
 ra[0]= (long) fmod(16807.0*(double)seed, 2147483647.0);
 for(i=1; i<=M; i++)
 {
  ra[i] = (long)fmod( 16807.0 * (double) ra[i-1], 2147483647.0);
 }
}

long randl(long num)      // random number between 0 and num-1
{
 return(RandomInteger % num);
}

double randd(void)
{
 return((double) RandomInteger / RIMAX);
}


#define VERBOSE (1)

#define N (50)  // number of neurons
#define SPR (0.2) //sparseness
#define OPTNUMMAX (100) // maximum number of patterns to be learnt
#define SLOPE (25) // slope of tanh(.)
#define NOUP (150) // number of update steps
#define SEED (42)


int OPTNUM;
int block; // number of 1s in a pattern
int opn; // maximal number of target-embryo pairs
  
int maxfitnum;
double output[N],temp,maxtemp;
double weight[N][N];
unsigned long int zeed = 0;
double opt[OPTNUMMAX][N];
double embrvect[N],sumfit,maxfit;
double rembr[OPTNUMMAX][N];
long idu;
double kezdfit,vegfit;

double fitn;

// the Pearson product-moment correlation coefficient 
double pear_correl(double* in, double* ou)
{
  double r;
  int m;
  double avinp, avres;
  double cov, sdinp, sdres;

  avinp=0.0;
  avres=0.0;
  
  for(m=0;m<N;m++)
  {
     avinp+=in[m];
     avres+=ou[m];
  }
  
  avinp=avinp/(double)N;
  avres=avres/(double)N;
 
  cov=0.0;
  sdinp=0.0;
  sdres=0.0;

  for(m=0;m<N;m++)
  {
    cov+=(in[m]-avinp)*(ou[m]-avres);
    sdinp+=(in[m]-avinp)*(in[m]-avinp);
    sdres+=(ou[m]-avres)*(ou[m]-avres);
  }
  
  r=cov/sqrt(sdinp)/sqrt(sdres);
  
  if(sdinp==0.0)
      r=-99999;
  if(sdres==0.0)
      r=-99999;
  
  return r; 
  
}

void init(void)
{
 
  int k,i,j;
  double max=0.0;
  
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      weight[i][j]=0.0;


  for(k=0;k<OPTNUM;k++)
  {
    for(i=0;i<N;i++)
    {
        opt[k][i]=0.0;
        rembr[k][i]=0.0;
    }
  }


  for(k=0;k<OPTNUM;k++)
  {
    for(i=0;i<N;i++)
    {
      if(randd()<SPR)
        opt[k][i]=5.0;
      else
        opt[k][i]=0.0;
      
      if(randd()<SPR)
        rembr[k][i]=5.0;
      else
        rembr[k][i]=0.0;      
    }
  }

  for(k=0;k<OPTNUM;k++)
    for(i=0;i<N;i++)
      for(j=0;j<N;j++)
        weight[i][j]+=(2.0*opt[k][i]/5.0-1.0)*(opt[k][j]/5.0-(double)SPR+rembr[k][j]/5.0-(double)SPR);


  for(i=0;i<N;i++)
  {
    for(j=0;j<N;j++)
    {
      if(fabs(weight[i][j])>max)
        max=fabs(weight[i][j]);
    }
  }
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      weight[i][j]/=max;
 
}



void random_mutate_embrvector()
{
  int num;

  num=randl(N);

  embrvect[num]=-1.0*embrvect[num]+5.0;

}

void update_output(int n)
{
   int t;
   double h=0.0;

   for(t = 0; t < N; t++)
     h += weight[n][t] * output[t];

   h=0.5*(1.0+tanh(SLOPE*h));

   output[n] = 0.8*output[n]+h; // linear threshold
}

double train_network(int op)
{
  int i,n,t;


  for(i=0;i<N;i++)
    embrvect[i]=rembr[op][i];

  for(n=0;n<N;n++)
    output[n]=embrvect[n];

  for(n=0;n<NOUP;n++)
    for(t=0;t<N;t++)
      update_output(t);

  fitn=pear_correl(output,opt[op]);
      
  return(fitn);
    
}


int main(int argc, char** argv)
{
  int opti,cou,num,nonsucc;
  double er,temp;
  double h,perf;
  seed(SEED);
  idu=(-1*SEED);
  

  for(num=1;num<=100;num++)
  {
    OPTNUM=num;
    perf=0.0;
    for(cou=0;cou<500;cou++)
    {
      init();
      er=0.0;
      nonsucc=0;
      for(opti=0;opti<OPTNUM;opti++)
      {
        temp=train_network(opti);
        if(temp!=-99999)
          er+=temp;
        else
          nonsucc++;
      }
      perf+=er/(double)(OPTNUM-nonsucc);        
      fflush(stdout);
    }
    h=perf/((double)cou);
    printf("%d\t%lf\n",OPTNUM,h);
    fflush(stdout);
  }

  return(0);
}
