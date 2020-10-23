
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <nrutil.h>
#include <nr.h>	

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

long randl(long num)      /* random number between 0 and num-1 */
{
 return(RandomInteger % num);
}

double randd(void)
{
 return((double) RandomInteger / RIMAX);
}

#define VERBOSE (1)

#define N (100)  // number of neurons
#define NN (100) // number of networks
#define OPTNUM (3) // number of patterns to be learnt
#define ACCURACY (0.95) // per bit relative accuracy criterion for termination
#define SLOPE (25) // slope of tanh(.)
#define VARW (0.05) // mutation interval of W
#define VARE (0.1) // mutation interval of G
#define NOUP (50) // number of update steps
#define SIGMA (0.1)
#define SEED (96561011)
#define EXP3 (4)
#define EXP2 (2)
#define EXP1 (2)
#define TRAINNUM (10000) // number of training before switching between patterns

int maxfitnum;
float output[NN][N],temp,maxtemp;
float weight[NN][N][N];
unsigned long int zeed = 0;
float embrvect[NN][N],sumfit,maxfit;
long idu;
float kezdfit,vegfit;

float rembr[3][100]={{0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,5,0,0,5,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,5,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,5,5,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,5,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,5},
{0,0,0,0,5,0,0,0,0,0,0,0,5,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,5,0,0,0,5,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0}};

float opt[3][100]={{0,0,0,0,0,5,0,0,0,0,0,5,0,0,0,0,0,0,0,5,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,5,0,5,0,0,0,5,0,0,0,0,0,0,0,0},
{0,0,0,0,0,5,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,5,0,5,0,0,0,0,5,5,0,0,0,5,0,0,0,0},
{0,0,0,0,0,5,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,5,0,0,0,0,5,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,5,0,0,0,0,0,0,5,5,0,0,0,0,0,0,0,0}};

//4 triple, 1 double, 3 single

float fitn[NN];

void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

void init(void)
{
  int i,j,k;


  for(k=0;k<NN;k++)
    for(i=0;i<N;i++)
      for(j=0;j<N;j++)
        weight[k][i][j]=0.0;

}
  
  
  
void draw_wmatr(int nu)
{
  int m,n;
  char str[255],str2[255];
  FILE *outm;//, *outf;

  snprintf(str,sizeof(str),"m.%03d",nu);
  snprintf(str2,sizeof(str2),"%03d.gif",nu);
  outm=fopen(str,"wt");

  for(m=0;m<N;m++)
  {
    for(n=0;n<N;n++)
      fprintf(outm,"%f\t",weight[maxfitnum][m][n]);
    fprintf(outm,"\n");
  }
  fclose(outm);

/*
  outf=fopen("file.gnuplot","wt");  
  fprintf(outf,"unset key\n");
  fprintf(outf,"set palette defined (0 \"blue\", 1 \"red\")\n");
  fprintf(outf,"set cbrange [-1:1]\n");
  fprintf(outf,"set terminal gif\n");
  fprintf(outf,"set output \"%s\"\n",str2);
  fprintf(outf,"plot \"%s\" matrix with image\n",str);
  fclose(outf);
  system("gnuplot file.gnuplot"); 
*/

return;
}


void random_mutate_network(int m)
{
  int i,j;

  i=randl(N);
  j=randl(N);

  weight[m][i][j]+=VARW*gasdev(&idu);

  if(weight[m][i][j]<-1.0)
    weight[m][i][j]=-1.0;
  if(weight[m][i][j]>1.0)
    weight[m][i][j]=1.0;
}

void random_mutate_embrvector(int m)
{
  int num;

  num=randl(N);

  embrvect[m][num]+=VARE*gasdev(&idu);

  if(embrvect[m][num]>5.0)
    embrvect[m][num]=5.0;
  if(embrvect[m][num]<0.0)
    embrvect[m][num]=0.0;

}


float fitness(float* in, float* ou)
{
  int t;
  float hd=0.0;

  for(t=0;t<N;t++)
    hd+=pow((in[t]-ou[t])/5.0,2.0);
  hd=-1.0*sqrt(hd/(float)N)+1.0;
    
  return(hd);
}


void update_output(int m, int n)
{
   int t;
   float h=0.0;

   for(t = 0; t < N; t++)
     h += weight[m][n][t] * output[m][t];

   h=0.5*(1.0+tanh(SLOPE*h));

   output[m][n] = 0.8*output[m][n]+h;

}

int train_network(int op)
{
  int rou,m,i,j,n,t;
  int pick;
  int c=0;

  kezdfit=0.0;
  vegfit=0.0;


  for(rou=1;rou<TRAINNUM;rou++)
  {
    sumfit=0.0;
    maxfit=0.0;
    maxfitnum=-99;

    for(j=0;j<NN;j++)
      for(i=0;i<N;i++)
        embrvect[j][i]=rembr[op][i];

    for(m=0;m<NN;m++) 
    {

      random_mutate_network(m);
      random_mutate_embrvector(m);


      for(n=0;n<N;n++)
        output[m][n]=embrvect[m][n];

      for(n=0;n<NOUP;n++)
        for(t=0;t<N;t++)
          update_output(m, randl(N));

      fitn[m]=fitness(output[m],opt[op]);

      sumfit+=fitn[m];
      if(fitn[m]>maxfit)
      {
        maxfit=fitn[m];
        maxfitnum=m;
      }
    } 

    m=maxfitnum;
    pick=randl(NN);
    while(pick==m)
      pick=randl(NN);

    for(i=0;i<N;i++)
    {
      for(j=0;j<N;j++)
        weight[pick][i][j]=weight[m][i][j];
    }

    if(rou==1)
      kezdfit=sumfit/(float)NN;

    if(sumfit/(float)NN>ACCURACY)
    {
      c++;
      if(c==20)
      {
        vegfit=sumfit/(float)NN;
        return(rou);
      }

    }
    else
      c=0;

  }

  vegfit=sumfit/(float)NN;
  return(rou);
}


int all_OK()
{
  int i,num;
  
  num=0;  
  for(i=0;i<OPTNUM;i++)
    num+=train_network(i);
  return(num);
}

int main(int argc, char** argv)
{

  int numneed,opti,num,succ,cou;
  seed(SEED);
  idu=(-1*SEED);

  succ=1;
  cou=0;
  init();

  while(succ<=100)
  {
     cou++;
     fflush(stdout);
     opti=randl(OPTNUM);
     numneed=train_network(opti);
     printf("%d\t%d\t%d\n",cou,opti,numneed);
     if(numneed==20)
     {
        num=all_OK();
        if(num==60)
        {
          printf("SUCCESS\n");
          fflush(stdout);
          draw_wmatr(succ);
          succ++;
          init();
        }
     }

  }

  return(0);
}
