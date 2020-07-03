#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include <cmath>

/* Este codigo gera N particulas numa caixa de tamanho L(=1), que interagem a partir do 
potencial de Lennard-Jones. As posicoes iniciais seguem uma distribuicao uniforme
e as velocidades iniciais uma distribuicao normal.

- Metodo de Euler
- Comparacao da forca que todas as particulas exercem umas nas outras
*/

struct configuracao {
  const unsigned          N;  // Numero de particulas
  double                  T;  // Temperatura
  double                  m;  // massa de cada particula
  double                  t;  // tempo total de interacao
  double                  L;  // tamanho da caixa
  double            delta_t;  // intervalo de tempo para a utilizacao do metodo de Euler
  double                sig;  // sigma, epsilon fatores do potencial de LJ
  double                eps;
  Eigen::Array<double, -1, -1> rx;         // (N,1) array da coordenada x das posicoes de cada partícula 
  Eigen::Array<double, -1, -1> ry;  
  Eigen::Array<double, -1, -1> vx;         // (N,1) array da coordenada x das velocidades de cada partícula 
  Eigen::Array<double, -1, -1> vy;
  Eigen::Array<double, -1, -1> rx0;        // Analogos aos de cima para guardas as posicoes em cada instante
  Eigen::Array<double, -1, -1> ry0;        //tempo para comparacao do potencial entre as particulas num momento estatico
  Eigen::Array<double, -1, -1> vx0;
  Eigen::Array<double, -1, -1> vy0;
  Eigen::Array<double, -1, -1> rx_total;   // (t,N) array que guarda as posições de cada uma das N particulas em cada instante
  Eigen::Array<double, -1, -1> ry_total;   //de tempo t considerado para plot da evolucao do sistema
  std::mt19937 rnd;
  std::normal_distribution<double> normal_dist{0., 1. };
  std::uniform_real_distribution<double> uniform_dist{0., 1.};
  
  configuracao(unsigned nn, double t1, double m1, int t2, double l1, double dt, double sg, double ep) : 
  N(nn), T(t1), m(m1), t(t2), L(l1), delta_t(dt), sig(sg), eps(ep),
  rx(nn,1), ry(nn,1), vx(nn,1), vy(nn,1),
  rx0(nn,1), ry0(nn,1), vx0(nn,1), vy0(nn,1),
  rx_total(t2,nn), ry_total(t2,nn)
  {
    std::random_device r;
    std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
    rnd.seed(seed2); 
  }
  
  inline double get_gaussian()
  {
    return normal_dist(rnd);
  }
  
  inline double get_uniform()
  {
    return uniform_dist(rnd);
  }

  double ljones(double sig, double eps, double r) //funcao que da a forca de lennard jones
  {
    double u;
    if (r != 0)  
    {
      u = 24. * eps * (2 * std::pow((sig/r),12) - std::pow(sig/r,6)) / r;
      u /= r; //divisao por r por causa do versor para converter as aleceracoes em ax,ay
    }
    else if (r == 0)  //retorna zero se estivermos a considerar a interacao da particula com ela propria
    {
      u = 0;
    }
    return u;
  }

  double fs(double r) //outros potenciais experimentados para testar o programa
  {
    double u;
    if (r != 0)
    {
      //u = 1 / (r + 1);
      u = 0.5 * std::pow(r,2);
      u /= r;
    }
    else if (r == 0)
    {
      u = 0;
    }
    
    return u;
  }


  void init_positions() //gera posicoes com uma distribuicao uniforme entr 0 e 1
  {
    
    for(unsigned i = 0; i < N; i++)
      {
	      rx(i) = get_uniform();
	      ry(i) = get_uniform();
      }  
  }
  
  
  void init_velocities() //gera velocidades com distribuicao gaussiana
  {
    
    for(unsigned i = 0; i < N; i++)
      {
      	vx(i) = sqrt(T/m)*get_gaussian();
	      vy(i) = sqrt(T/m)*get_gaussian();
      }  

  }
  
  void return_positions() //funcao principal
  {
    init_positions();
    init_velocities();
    for(unsigned i = 0; i < t; i++) //ciclo sobre todos os tempos
    {
      rx0 = rx;  //arrays que registam as posicoes e velocidades das particulas em cada tempo para
      ry0 = ry;  //a comparacao entre elas ser num tempo fixo
      vx0 = vx;
      vy0 = vy;

      for(unsigned j = 0; j < N; j++) //ciclo sobre todas as particulas
      {

        rx_total(i,j) = rx(j); //preenche um array com os tempos como linhas e posicao das particulas em colunas para registo total
        ry_total(i,j) = ry(j);
        
        double x_temp;
        double y_temp;
        double ax_temp = 0; //aceleracoes definidas zero no inicio de cada ciclo para se poder fazer a soma dos efeitos de todas
        double ay_temp = 0;
        double r_temp;
        double lj = 0;


        for(unsigned k = 0; k < N; k++)
        {
          r_temp = std::pow(std::pow((rx0(k) - rx0(j)),2) + std::pow((ry0(k) - ry0(j)),2),0.5); //distancia entre as particulas
          //lj = fs(r_temp);
          lj = ljones(sig,eps,r_temp);
          ax_temp += lj / m * (rx0(k) - rx0(j)); //forca a dividir pela massa vezes o restante do versor
          ay_temp += lj / m * (ry0(k) - ry0(j));
        }
        x_temp = rx(j) + vx(j) * delta_t + 0.5 * ax_temp * std::pow(delta_t,2); //posicoes propostas com metodo de Euler
        y_temp = ry(j) + vy(j) * delta_t + 0.5 * ay_temp * std::pow(delta_t,2);

        if (x_temp > L || x_temp < 0)                //condicoes fronteira para saida da caixa em x
        {
          vx(j) *= -1;                               //inversao da velocidade
          if (x_temp > L)
          {
            rx(j) = L - (x_temp - L);                //posicao refletida se sair pelo lado direito
          }

          else if (x_temp < 0)                    
          {
            rx(j) = x_temp * -1;                     //posicao refletida se sair pelo lado esquerdo
          }
        }
        else if (x_temp <= L && x_temp >= 0)         //condicao para a particula se manter na caixa sem interagir com a fronteira
        {
          rx(j) = x_temp;                            //aceitacao do valor proposto
        }



        if (y_temp > L || y_temp < 0)                //condicoes fronteira para a saida da caixa em y analogo a x
        { 
          vy(j) *= -1;
          if (y_temp > L)
          {
            ry(j) = L - (y_temp - L);
          }

          else if (y_temp < 0)
          {
            ry(j) = y_temp * -1;
          }
        }
        else if (y_temp <= L && y_temp >= 0){
          ry(j) = y_temp;
        }




      vx(j) += ax_temp * delta_t;          //atualizacao das velocidades
      vy(j) += ay_temp * delta_t;

      }
      
    }
    //std::cout << vx0 << "\n\n";
    //std::cout << vy0 << "\n\n";

    std::ofstream fx("rxlj.txt"); //escrita do array (t,N) para as posicoes em x num ficheiro de texto
    if (fx.is_open()){
      fx << rx_total;
    }
    else    {
      std::cout << "Could not open file" << "\n";
    }
    
    fx.close();




    std::ofstream fy("rylj.txt");  //escrita do array (t,N) para as posicoes em y num ficheiro de texto
    if (fy.is_open()){
      fy << ry_total;
    }
    else    {
      std::cout << "Could not open file" << "\n";
    }
    
    fy.close();

  }
  
};



int main()
{
  const unsigned                              N = 25;   // Numero de particulas
  double                                    T = 200;   // Temperatura
  double                   m = 1.* std::pow(10,-19) ;   // massa de cada particula
  double                                  t = 50000.;   // tempo total de interacao
  double                                     L = 1. ;   // tamanho da caixa
  double             delta_t = 0.1 * std::pow(10,-3);   // intervalo de tempo considerado para o metodo de Euler
  double                                  sig = 0.01;   // sigma e epsilon fatores do potencial de LJ
  double                                  eps = 0.01;
  configuracao c1(N,T,m,t,L,delta_t,sig,eps);
  c1.return_positions();
}
