#include "ran2.c"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


long seed = 904799957;

// Funciones referente a las listas
// Definicion de la lista
typedef struct Node {
  int IDent;
  struct Node *pNext;
} * List_t;
#define LIST_EMPTY NULL
#define LIST_IDent(lst) ((lst)->IDent)
#define LIST_TAIL(lst) ((lst)->pNext)
// Insertar nodos en la lista
List_t List_Node_Create(int e, List_t tail) {
  List_t newL = (List_t)malloc(sizeof(struct Node));
  assert(newL);
  LIST_IDent(newL) = e;
  LIST_TAIL(newL) = tail;
  return newL;
}

// Definici�n de las estructuras a emplear
// padres
typedef struct {
  int ID_par;
  // int ID_Padre;
  // int ID_madre;
  List_t Sons;      // lista con los hijos varones
  List_t Daughters; // lista con hijas
} Node_Parents;
// nodos hijos
typedef struct {
  char sex;     // define el sexo del nodo
  int ID;       // n�mero del nodo
  int parents;  // definir pareja de padres, esto ser�n n�meros de 0 a N-1
  _Bool engage; //*engaged         //esta variable dice si esta casado o no la
                //persona (nodo). Es un Bool, asi que sera 0 si la persona es
                // soltera y sera '1' si esta casada
  int marriage; // ID de la pareja  (esto quizas cambie en conjunto con la
                // variable "engage")
} Node_Person;
// conjunto de todos(red completa)
typedef struct {
  Node_Parents *P;
  Node_Person *Males;
  Node_Person *Females;
} Net;

// Creaci�n de la red:

// Funci�n que crea un array de N parejas de padres enumeradas de 0 a N-1.
// Las listas que contienen a los hijos estaran vacias
Node_Parents *allocaParents(int N) {
  Node_Parents *Par = (Node_Parents *)malloc(N * sizeof(Node_Parents));
  assert(Par != NULL);
  for (int i = 0; i < N; i++) // asignaci�n de identidad a las parejas de padres
  {
    Par[i].ID_par = i;
    Par[i].Daughters = LIST_EMPTY;
    Par[i].Sons = LIST_EMPTY;
  }
  return Par;
}
// Funci�n que crea un array de N nodos, enumerados de 0 a N-1. Puede crearlos
// masculinos o femeninos en dependencia del par�metro que uno ingrese. Esta
// funci�n tambi�n asigna la pareja de padres i-�sima al nodo i-�simo. **Esta
// informacion no se cargar� en la lista de Hijos de la pareja de padres.**
Node_Person *allocaPersons(int N, char c) {
  Node_Person *Pers = (Node_Person *)malloc(N * sizeof(Node_Person));
  assert(Pers != NULL);
  for (int i = 0; i < N; i++) {
    Pers[i].ID = i;
    Pers[i].parents = i;
    Pers[i].sex = c;
    Pers[i].engage = 0;
  }
  return Pers;
}
// Funcion que crea la red
Net allocateNet(int N) {
  char M = 'M';
  char F = 'F';
  Net Red; // se crea una red generica y a continuacion se alloca la memoria de
           // sus elementos
  Red.P = allocaParents(N);
  Red.Females = allocaPersons(N, F);
  Red.Males = allocaPersons(N, M);
  return Red;
}

int Rand_Gen(int N) // Funcion que genera un numero aleatorio ENTERO entre a y b
{
  return (int)(ran2(&seed) * (float)(N));
}

/*Funci�n que reasigna los padres a la red. Falta a�adir las l�neas referentes
a cuando se asigna un hijo, anotarlo en una lista para los padres*/
void reasign_parents(Net Red, double rho, int N) {
  double r; // numero real aleatorio entre 0 y 1, abajo se calcula
  int A;    // entero que me va a decir a que pareja de padres enviarlo
  for (int i = 0; i < N; i++) {
    r = (double)ran2(&seed);
    // eventualmente cambiar la generacion de numeros aleatorios, no solo con el
    // rand()
    if (r < rho) {
      A = Rand_Gen(N);
      /*while((Red.Males[i].parents)==A)
          A=Rand_Gen(N);*/
      Red.Males[i].parents = A;
    }
    r = (double)ran2(&seed);
    // printf("%g\n", r);
    if (r < rho) {
      A = Rand_Gen(N);
      /*while((Red.Females[i].parents)==A)
          A=Rand_Gen(N);*/
      Red.Females[i].parents = A;
    }
  }

  for (int i = 0; i < N; i++) {
    Red.P[Red.Males[i].parents].Sons =
        List_Node_Create(i, Red.P[Red.Males[i].parents].Sons);
    Red.P[Red.Females[i].parents].Daughters =
        List_Node_Create(i, Red.P[Red.Females[i].parents].Daughters);
  }
}

// Asignaci�n de lazos matrimoniales
int Marriages(Net Red, int N) {
  int A, cont;
  for (int i = 0; i < N; i++) {
    if (Red.Males[i].engage == 0) {
      A = Rand_Gen(N);
      cont = 0;
      while (Red.Males[i].parents == Red.Females[A].parents ||
             Red.Females[A].engage != 0) {
        A = Rand_Gen(N);
        cont++;
        if (cont ==
            100 * N) // esta parte es para el caso especial comentado debajo
        {
          for (int j = 0; j < N; j++) {
            Red.Males[j].engage = 0;
            Red.Females[j].engage = 0;
          }
          return 0; // con esto salgo de la funcion directamente y regreso un
                    // cero, que se interpretar� como que no se pudo realizar la
                    // asignaci�n de parejas y es necesario volverlo a hacer
        }
      }
      Red.Males[i].engage = 1;
      Red.Females[A].engage = 1;
      Red.Males[i].marriage = A;
      Red.Females[A].marriage = i;
    }
  }
  return 1;
}

/*
    Se puede encontrar el problea que Puede que el ultimo nodo Masculino solo
   tenga disponible una hermana para casarse, hacer un filtro para este caso.
   Una opci�n es volver a hacer la asignaci�n. Otro es elegir otro al azar y
   hacer un cambio de esposas
*/

// Funcion auxiliar para visualizar el parentesco padres-hijos en pantalla
void print_Net(Net Red, int N) {
  for (int i = 0; i < N; i++) {
    printf("%d %d m \n", i, Red.Males[i].marriage);
  }
}
// funcion auxiliar que imprime los elementos de una lista
void List_Print(List_t lst) // funcion que imprime una lista
{
  while (lst) {
    printf("%d ", LIST_IDent(lst));
    lst = LIST_TAIL(lst); // lst=lst->pNext;
  }
  puts(" ");
}
// Funcion auxiliar para imprimir los matrimonios en pantalla
void Print_Kinship(Net Red, int N) {
  printf("\n");
  List_t lst;
  int aux;
  for (int i = 0; i < N; i++) {
    lst = Red.P[Red.Males[i].parents].Sons;
    printf("El hombre %d es hijo de %d\n", i, Red.Males[i].parents);
    while (lst) {
      if (i == LIST_IDent(lst))
        lst = LIST_TAIL(lst);
      else {
        printf("El hombre %d es hermano del hombre %d \n", i, LIST_IDent(lst));
        lst = LIST_TAIL(lst);
      }
    }
    printf("El hombre %d  esta casado con la mujer %d\n", i,
           Red.Males[i].marriage);
    printf("Los suegros de� hombre %d son %d\n", i,
           Red.Females[Red.Males[i].marriage].parents);
    lst = Red.P[Red.Females[Red.Males[i].marriage].parents].Sons;
    while (lst) {
      printf("El hombre %d es cu�ado del hombre %d \n", i, LIST_IDent(lst));
      lst = LIST_TAIL(lst);
    }
  }
}

// Exportaci�n de las relaciones de parentesco en un archivo e impresion en la
// pantalla (esto ultimo se puede omitir comentando las lineas
void Export_Kinship(Net Red, int N, FILE *f) {
  List_t lst;
  for (int i = 0; i < N; i++) {
    lst = Red.P[Red.Males[i].parents].Sons;
    while (lst) {
      if (i == LIST_IDent(lst) ||
          LIST_IDent(lst) < i) // esto omite que una persona sea clasificada
                               // como su propio hermano
        lst = LIST_TAIL(lst);
      else {
        // printf("%d\t%d\th\n", i, LIST_IDent(lst));
        fprintf(f, "%d\t%d\t0\n", i, LIST_IDent(lst));
        // fprintf(f, "%d->%d,", i, LIST_IDent(lst));

        lst = LIST_TAIL(lst);
      }
    }
    lst = Red.P[Red.Females[Red.Males[i].marriage].parents].Sons;
    while (lst) {
      // printf("%d\t%d\tc1\n", i, LIST_IDent(lst));
      fprintf(f, "%d\t%d\t1\n", i, LIST_IDent(lst));
      // fprintf(f, "%d->%d,", i, LIST_IDent(lst));
      lst = LIST_TAIL(lst);
    }
    /*lst=Red.P[Red.Males[i].parents].Daughters;
    while(lst){
        if(Red.Females[LIST_IDent(lst)].engage==1)
        {
            //printf("%d\t%d\tc2\n", i, Red.Females[LIST_IDent(lst)].marriage);
            fprintf(f, "%d\t%d\t2\n", i, Red.Females[LIST_IDent(lst)].marriage);
            //fprintf(f, "%d->%d,", i, Red.Females[LIST_IDent(lst)].marriage);
        }
        lst=LIST_TAIL(lst);
    }*/
  }
  /*
  Exportar los casos especiales de ningun hermano o hermana, y esa persona
  casada con una mujer sin hermanos
  */
}

// Funciones para liberar la Red de la memoria
void Free_List(List_t lst) {
  if (lst != LIST_EMPTY)
    Free_List(LIST_TAIL(lst));
  free(lst);
}

void Free_Net(int N, Net Red) {
  for (int i = 0; i < N; i++) {
    Free_List(Red.P[i].Daughters);
    Free_List(Red.P[i].Sons);
  }
  free(Red.P);
  free(Red.Females);
  free(Red.Males);
}

void Iteracion(double rho, int N, char *File_Name, int M) {
  Net Network;
  printf("El programa ha iniciado\n");

  FILE *f;
  f = fopen(File_Name, "w");
  assert(f);
  int sucess;

  for (int i = 0; i < M; i++) {
    // printf("%d", i+1);
    Network = allocateNet(N);
    // printf("\t1");
    reasign_parents(Network, rho, N);
    // printf("\t2");
    sucess = Marriages(Network, N);
    while (sucess == 0) {
      sucess = Marriages(Network, N);
      printf("\nHizo falta volver a asignar matrimonios, caso especial\n");
    }
    // printf("\t3");
    Print_Kinship(Network, N);
    Export_Kinship(Network, N, f);
    // printf("\t4");
    Free_Net(N, Network);
    // printf("\n");
    fprintf(f, "0\t0\t0\n");
    // printf("\t5\n");
    // printf("Se ha completado el ciclo %d\n", i+1);
  }

  printf(
      "Ha terminado de guardarse en el archivo .txt\nrho=%g\tN=%d\t%d veces\n",
      rho, N, M);

  // printf("2 Se ha creado la red con la condicion inicial y N=%d\n", N);

  // printf("3 Se han reasignado los padres de acuerdo con el rho=%g\n", rho);

  // printf("4 Se han asignado matrimonios entre los nodos\n");

  // printf("5 Se ha exportado la red a un .csv \n");

  // printf("6 Se ha liberado la memoria utilizada en la ejecucion\n");
  fclose(f);
}

int main() {
  srand(time(NULL)); // Inicializa generador pseudo-aleatorio (no se si sea
                     // necesario en la librer�a rand2.c)
  // clock_t start, stop;

  int N = 7;         // Cantidad de hijos hombres en la red
  int M = 1;         // Cantidad de veces que se va a generar la red
  double rho = 0.70; // Par�metro rho

  /*for(int i=0; i<10*N; i++)
  {
      printf("%d\n", Rand_Gen(N));
  }*/
  Iteracion(rho, N, "Modelo.txt", M);

  /*FILE *f;      //Esto lo hice para medir los tiempos de ejecuci�n que
  demoraba f=fopen("tiempos.txt", "w");

  for(int i=0; i<30; i++)
  {
      start=clock();
      Iteracion(rho, N, "Prueba07.txt");
      stop=clock();
      fprintf(f, "%g\n", (double)(stop-start)/CLOCKS_PER_SEC);
  }
  fclose(f);*/

  // printf("\n%g\n", (double)(stop-start)/CLOCKS_PER_SEC);
  /*for(int i=0; i<N; i++)
  {
      printf("%g\n", ran2(&seed));
  }*/
  return 0;
}
