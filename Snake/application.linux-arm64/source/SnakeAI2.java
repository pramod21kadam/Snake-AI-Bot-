import processing.core.*; 
import processing.data.*; 
import processing.event.*; 
import processing.opengl.*; 

import java.util.HashMap; 
import java.util.ArrayList; 
import java.io.File; 
import java.io.BufferedReader; 
import java.io.PrintWriter; 
import java.io.InputStream; 
import java.io.OutputStream; 
import java.io.IOException; 

public class SnakeAI2 extends PApplet {

final int SIZE = 20;
final int hidden_nodes = 16;
final int hidden_layers = 2;
final int fps = 1500;  //15 is ideal for self play, increasing for AI does not directly increase speed, speed is dependant on processing power

int highscore = 0;
int score=0;
float mutationRate = 0.05f;
float defaultmutation = mutationRate;
  //false for AI, true to play yourself
boolean replayBest = true;  //shows only the best of each generation
boolean seeVision = false;  //see the snakes vision

int generation = 0;
ArrayList<Integer> evolution;

boolean show = false;

Population pop;
public void settings() {
  size(1000,1000);
}

public void setup() {
  evolution = new ArrayList<Integer>();
  frameRate(fps);
  pop = new Population(2000);
}

public void draw() {
  background(255);
                      if(pop.done()) {
                          highscore = pop.bestSnake.score;
                          pop.calculateFitness();
                          pop.naturalSelection();
                          score = highscore;
                          generation++;
                        } else {
                          pop.update();
                          if (!show)
                              pop.show();
                          fill(127);
                          stroke(127);
                          textSize(20);
                          text("High SCORE : "+ score+ "     Mutation Rate: " + mutationRate + "\nGeneration : " + generation ,50 ,50);
                        }               
  //grid();  
}
  
  public void grid(){
    fill(127);
      for(int i=0 ; i<50;i++){
        line(i*20,0,i*20,height);
        line(0,i*20,width,i*20);
      }
  }
  
  public void keyPressed(){
    if (keyCode == UP )
          mutationRate += 0.05f;
    if (keyCode == DOWN && mutationRate > 0.25f )
          mutationRate -= 0.05f;
  
  }
class Food {
    PVector pos;
    
    Food() {
      int x = 400 + SIZE + floor(random(20))*SIZE;
      int y = SIZE + floor(random(20))*SIZE;
      pos = new PVector(x,y);
    }
    
    public void show() {
       stroke(0);
       fill(255,0,0);
       rect(pos.x,pos.y,SIZE,SIZE);
    }
    
    public Food clone() {
       Food clone = new Food();
       clone.pos.x = pos.x;
       clone.pos.y = pos.y;
       return clone;
    }
}
class Matrix {
  
  int rows, cols;
  float[][] matrix;
  
   Matrix(int r, int c) {
     rows = r;
     cols = c;
     matrix = new float[rows][cols];
   }
   
   Matrix(float[][] m) {
      matrix = m;
      rows = matrix.length;
      cols = matrix[0].length;
      output();
   }
   
   public void output() {
      for(int i = 0; i < rows; i++) {
         for(int j = 0; j < cols; j++) {
            print(matrix[i][j] + " "); 
         }
         println();
      }
      println();
   }
   
   public Matrix dot(Matrix n) {
     Matrix result = new Matrix(rows, n.cols);
     
     if(cols == n.rows) {
        for(int i = 0; i < rows; i++) {
           for(int j = 0; j < n.cols; j++) {
              float sum = 0;
              for(int k = 0; k < cols; k++) {
                 sum += matrix[i][k]*n.matrix[k][j];
              }  
              result.matrix[i][j] = sum;
           }
        }
     }
     return result;
   }
   
   public void randomize() {
      for(int i = 0; i < rows; i++) {
         for(int j = 0; j < cols; j++) {
            matrix[i][j] = random(-1,1); 
         }
      }
   }
   
   public Matrix singleColumnMatrixFromArray(float[] arr) {
      Matrix n = new Matrix(arr.length, 1);
      for(int i = 0; i < arr.length; i++) {
         n.matrix[i][0] = arr[i]; 
      }
      return n;
   }
   
   public float[] toArray() {
      float[] arr = new float[rows*cols];
      for(int i = 0; i < rows; i++) {
         for(int j = 0; j < cols; j++) {
            arr[j+i*cols] = matrix[i][j]; 
         }
      }
      return arr;
   }
   
   public Matrix addBias() {
      Matrix n = new Matrix(rows+1, 1);
      for(int i = 0; i < rows; i++) {
         n.matrix[i][0] = matrix[i][0]; 
      }
      n.matrix[rows][0] = 1;
      return n;
   }
   
   public Matrix activate() {
      Matrix n = new Matrix(rows, cols);
      for(int i = 0; i < rows; i++) {
         for(int j = 0; j < cols; j++) {
            n.matrix[i][j] = relu(matrix[i][j]); 
         }
      }
      return n;
   }
   
   public float relu(float x) {
       return max(0,x);
   }
   
   public void mutate(float mutationRate) {
      for(int i = 0; i < rows; i++) {
         for(int j = 0; j < cols; j++) {
            float rand = random(1);
            if(rand<mutationRate) {
               matrix[i][j] += randomGaussian()/5;
               
               if(matrix[i][j] > 1) {
                  matrix[i][j] = 1;
               }
               if(matrix[i][j] <-1) {
                 matrix[i][j] = -1;
               }
            }
         }
      }
   }
   
   public Matrix crossover(Matrix partner) {
      Matrix child = new Matrix(rows, cols);
      
      int randC = floor(random(cols));
      int randR = floor(random(rows));
      
      for(int i = 0; i < rows; i++) {
         for(int j = 0;  j < cols; j++) {
            if((i  < randR) || (i == randR && j <= randC)) {
               child.matrix[i][j] = matrix[i][j]; 
            } else {
              child.matrix[i][j] = partner.matrix[i][j];
            }
         }
      }
      return child;
   }
   
   public Matrix clone() {
      Matrix clone = new Matrix(rows, cols);
      for(int i = 0; i < rows; i++) {
         for(int j = 0; j < cols; j++) {
            clone.matrix[i][j] = matrix[i][j]; 
         }
      }
      return clone;
   }
}
class NeuralNet {
  
  int iNodes, hNodes, oNodes, hLayers;
  Matrix[] weights;
  
  NeuralNet(int input, int hidden, int output, int hiddenLayers) {
    iNodes = input;
    hNodes = hidden;
    oNodes = output;
    hLayers = hiddenLayers;
    
    weights = new Matrix[hLayers+1];
    weights[0] = new Matrix(hNodes, iNodes+1);
    for(int i=1; i<hLayers; i++) {
       weights[i] = new Matrix(hNodes,hNodes+1); 
    }
    weights[weights.length-1] = new Matrix(oNodes,hNodes+1);
    
    for(Matrix w : weights) {
       w.randomize(); 
    }
  }
  
  public void mutate(float mr) {
     for(Matrix w : weights) {
        w.mutate(mr); 
     }
  }
  
  public float[] output(float[] inputsArr) {
     Matrix inputs = weights[0].singleColumnMatrixFromArray(inputsArr);
     
     Matrix curr_bias = inputs.addBias();
     
     for(int i=0; i<hLayers; i++) {
        Matrix hidden_ip = weights[i].dot(curr_bias); 
        Matrix hidden_op = hidden_ip.activate();
        curr_bias = hidden_op.addBias();
     }
     
     Matrix output_ip = weights[weights.length-1].dot(curr_bias);
     Matrix output = output_ip.activate();
     
     return output.toArray();
  }
  
  public NeuralNet crossover(NeuralNet partner) {
     NeuralNet child = new NeuralNet(iNodes,hNodes,oNodes,hLayers);
     for(int i=0; i<weights.length; i++) {
        child.weights[i] = weights[i].crossover(partner.weights[i]);
     }
     return child;
  }
  
  public NeuralNet clone() {
     NeuralNet clone = new NeuralNet(iNodes,hNodes,oNodes,hLayers);
     for(int i=0; i<weights.length; i++) {
        clone.weights[i] = weights[i].clone(); 
     }
     
     return clone;
  }
  
  public void load(Matrix[] weight) {
      for(int i=0; i<weights.length; i++) {
         weights[i] = weight[i]; 
      }
  }
  
  public Matrix[] pull() {
     Matrix[] model = weights.clone();
     return model;
  }
}
class Population {
   
   Snake[] snakes;
   Snake bestSnake;
   
   int bestSnakeScore = 0;
   int gen = 0;
   int samebest = 0;
   
   float bestFitness = 0;
   float fitnessSum = 0;
   
   Population(int size) {
      snakes = new Snake[size]; 
      for(int i = 0; i < snakes.length; i++) {
         snakes[i] = new Snake(); 
      }
      bestSnake = snakes[0].clone();
      bestSnake.replay = true;
   }
   
   public boolean done() {  //check if all the snakes in the population are dead
      for(int i = 0; i < snakes.length; i++) {
         if(!snakes[i].dead)
           return false;
      }
      if(!bestSnake.dead) {
         return false; 
      }
      return true;
   }
   
   public void update() {  //update all the snakes in the generation
      if(!bestSnake.dead) {  //if the best snake is not dead update it, this snake is a replay of the best from the past generation
         bestSnake.look();
         bestSnake.think();
         bestSnake.move();
      }
      for(int i = 0; i < snakes.length; i++) {
        if(!snakes[i].dead) {
           snakes[i].look();
           snakes[i].think();
           snakes[i].move(); 
        }
      }
   }
   
   public void show() {  //show either the best snake or all the snakes
      if(replayBest) {
        bestSnake.show();
      } else {
         for(int i = 0; i < snakes.length; i++) {
            snakes[i].show(); 
         }
      }
   }
   
   public void setBestSnake() {  //set the best snake of the generation
       float max = 0;
       int maxIndex = 0;
       for(int i = 0; i < snakes.length; i++) {
          if(snakes[i].fitness > max) {
             max = snakes[i].fitness;
             maxIndex = i;
          }
       }
       if(max > bestFitness) {
         bestFitness = max;
         bestSnake = snakes[maxIndex].cloneForReplay();
         bestSnakeScore = snakes[maxIndex].score;
         //samebest = 0;
         //mutationRate = defaultMutation;
       } else {
         bestSnake = bestSnake.cloneForReplay(); 
         /*
         samebest++;
         if(samebest > 2) {  //if the best snake has remained the same for more than 3 generations, raise the mutation rate
            mutationRate *= 2;
            samebest = 0;
         }*/
       }
   }
   
   public Snake selectParent() {  //selects a random number in range of the fitnesssum and if a snake falls in that range then select it
      float rand = random(fitnessSum);
      float summation = 0;
      for(int i = 0; i < snakes.length; i++) {
         summation += snakes[i].fitness;
         if(summation > rand) {
           return snakes[i];
         }
      }
      return snakes[0];
   }
   
   public void naturalSelection() {
      Snake[] newSnakes = new Snake[snakes.length];
      
      setBestSnake();
      calculateFitnessSum();
      
      newSnakes[0] = bestSnake.clone();  //add the best snake of the prior generation into the new generation
      for(int i = 1; i < snakes.length; i++) {
         Snake child = selectParent().crossover(selectParent());
         child.mutate();
         newSnakes[i] = child;
      }
      snakes = newSnakes.clone();
      evolution.add(bestSnakeScore);
      gen+=1;
   }
   
   public void mutate() {
       for(int i = 1; i < snakes.length; i++) {  //start from 1 as to not override the best snake placed in index 0
          snakes[i].mutate(); 
       }
   }
   
   public void calculateFitness() {  //calculate the fitnesses for each snake
      for(int i = 0; i < snakes.length; i++) {
         snakes[i].calculateFitness(); 
      }
   }
   
   public void calculateFitnessSum() {  //calculate the sum of all the snakes fitnesses
       fitnessSum = 0;
       for(int i = 0; i < snakes.length; i++) {
         fitnessSum += snakes[i].fitness; 
      }
   }
}
class Snake {
   
  int score = 0;
  int lifeLeft = 200;  //amount of moves the snake can make before it dies
  int lifetime = 0;  //amount of time the snake has been alive
  int xVel, yVel;
  int foodItterate = 0;  //itterator to run through the foodlist (used for replay)
  
  float fitness = 0;
  
  boolean dead = false;
  boolean replay = false;  //if this snake is a replay of best snake
  
  float[] vision;  //snakes vision
  float[] decision;  //snakes decision
  
  PVector head;
  
  ArrayList<PVector> body;  //snakes body
  ArrayList<Food> foodList;  //list of food positions (used to replay the best snake)
  
  Food food;
  NeuralNet brain;
  
  Snake() {
    this(hidden_layers);
  }
  
  Snake(int layers) {
    head = new PVector(00,height/2);
    food = new Food();
    body = new ArrayList<PVector>();

      vision = new float[24];
      decision = new float[4];
      foodList = new ArrayList<Food>();
      foodList.add(food.clone());
      brain = new NeuralNet(24,hidden_nodes,4,layers);
      body.add(new PVector(00,(height/2)+SIZE));  
      body.add(new PVector(00,(height/2)+(2*SIZE)));
      score+=2;
    }
  
  Snake(ArrayList<Food> foods) {  //this constructor passes in a list of food positions so that a replay can replay the best snake
     replay = true;
     vision = new float[24];
     decision = new float[4];
     body = new ArrayList<PVector>();
     foodList = new ArrayList<Food>(foods.size());
     for(Food f: foods) {  //clone all the food positions in the foodlist
       foodList.add(f.clone());
     }
     food = foodList.get(foodItterate);
     foodItterate++;
     head = new PVector(00,height/2);
     body.add(new PVector(00,(height/2)+SIZE));
     body.add(new PVector(00,(height/2)+(2*SIZE)));
     score += 2 ;
  }
  
  public boolean bodyCollide(float x, float y) {  //check if a position collides with the snakes body
     for(int i = 0; i < body.size(); i++) {
        if(x == body.get(i).x && y == body.get(i).y)  {
           return true;
        }
     }
     return false;
  }
  
  public boolean foodCollide(float x, float y) {  //check if a position collides with the food
     if(x == food.pos.x && y == food.pos.y) {
         return true;
     }
     return false;
  }
  
  public boolean wallCollide(float x, float y) {
    if(x >= width-(SIZE) || x < SIZE || y >= height-(SIZE) || y < SIZE) {
       return true;
     }
     return false;
  }
  
  public void show() {
     food.show();
     fill(127,80);
     stroke(127);
     for(int i = 0; i < body.size(); i++) {
       rect(body.get(i).x,body.get(i).y,SIZE,SIZE);
     }
     if(dead) {
       fill(127,80);
     } else {
       fill(127,80);
     }
     rect(head.x,head.y,SIZE,SIZE);
     fill(127);
     text("SCORE : "+ score ,50 ,125);
  }
  
  public void move() {  //move the snake
     if(!dead){
         lifetime++;
         lifeLeft--;
       if(foodCollide(head.x,head.y)) {
          eat();
       }
       shiftBody();
       if(wallCollide(head.x,head.y)) {
         dead = true;
       } else if(bodyCollide(head.x,head.y)) {
         dead = true;
       } else if(lifeLeft <= 0) {
          dead = true;
       }
     }
  }
  
  public void eat() {  //eat food
    int len = body.size()-1;
    score++;

      if(lifeLeft < 500) {
        if(lifeLeft > 400) {
           lifeLeft = 500; 
        } else {
          lifeLeft+=100;
        }
      }

    if(len >= 0) {
      body.add(new PVector(body.get(len).x,body.get(len).y));
    } else {
      body.add(new PVector(head.x,head.y)); 
    }
    if(!replay) {
      food = new Food();
      while(bodyCollide(food.pos.x,food.pos.y)) {
         food = new Food();
      }

        foodList.add(food);
    } else {  //if the snake is a replay, then we dont want to create new random foods, we want to see the positions the best snake had to collect
      food = foodList.get(foodItterate);
      foodItterate++;
    }
  }
  
  public void shiftBody() {  //shift the body to follow the head
    float tempx = head.x;
    float tempy = head.y;
    head.x += xVel;
    head.y += yVel;
    float temp2x;
    float temp2y;
    for(int i = 0; i < body.size(); i++) {
       temp2x = body.get(i).x;
       temp2y = body.get(i).y;
       body.get(i).x = tempx;
       body.get(i).y = tempy;
       tempx = temp2x;
       tempy = temp2y;
    } 
  }
  
  public Snake cloneForReplay() {  //clone a version of the snake that will be used for a replay
     Snake clone = new Snake(foodList);
     clone.brain = brain.clone();
     return clone;
  }
  
  public Snake clone() {  //clone the snake
     Snake clone = new Snake(hidden_layers);
     clone.brain = brain.clone();
     return clone;
  }
  
  public Snake crossover(Snake parent) {  //crossover the snake with another snake
     Snake child = new Snake(hidden_layers);
     child.brain = brain.crossover(parent.brain);
     return child;
  }
  
  public void mutate() {  //mutate the snakes brain
     brain.mutate(mutationRate); 
  }
  
  public void calculateFitness() {  //calculate the fitness of the snake
     if(score < 10) {
        fitness = floor(lifetime * lifetime) * pow(2,score); 
     } else {
        fitness = floor(lifetime * lifetime);
        fitness *= pow(2,10);
        fitness *= (score-9);
     }
  }
  
  public void look() {  //look in all 8 directions and check for food, body and wall
    vision = new float[24];
    float[] temp = lookInDirection(new PVector(-SIZE,0));
    vision[0] = temp[0];
    vision[1] = temp[1];
    vision[2] = temp[2];
    temp = lookInDirection(new PVector(-SIZE,-SIZE));
    vision[3] = temp[0];
    vision[4] = temp[1];
    vision[5] = temp[2];
    temp = lookInDirection(new PVector(0,-SIZE));
    vision[6] = temp[0];
    vision[7] = temp[1];
    vision[8] = temp[2];
    temp = lookInDirection(new PVector(SIZE,-SIZE));
    vision[9] = temp[0];
    vision[10] = temp[1];
    vision[11] = temp[2];
    temp = lookInDirection(new PVector(SIZE,0));
    vision[12] = temp[0];
    vision[13] = temp[1];
    vision[14] = temp[2];
    temp = lookInDirection(new PVector(SIZE,SIZE));
    vision[15] = temp[0];
    vision[16] = temp[1];
    vision[17] = temp[2];
    temp = lookInDirection(new PVector(0,SIZE));
    vision[18] = temp[0];
    vision[19] = temp[1];
    vision[20] = temp[2];
    temp = lookInDirection(new PVector(-SIZE,SIZE));
    vision[21] = temp[0];
    vision[22] = temp[1];
    vision[23] = temp[2];
  }

  public float[] lookInDirection(PVector direction) {  //look in a direction and check for food, body and wall
    float look[] = new float[3];
    PVector pos = new PVector(head.x,  head.y);
    float distance = 0;
    boolean foodFound = false;
    boolean bodyFound = false;
    pos.add(direction);
    distance +=1;
    while (!wallCollide(pos.x,pos.y)) {
      if(!foodFound && foodCollide(pos.x,pos.y)) {
        foodFound = true;
        look[0] = 1;
      }
      if(!bodyFound && bodyCollide(pos.x,pos.y)) {
         bodyFound = true;
         look[1] = 1;
      }
      if(replay && seeVision) {
        stroke(0,255,0);
        point(pos.x,pos.y);
        if(foodFound) {
           noStroke();
           fill(255,255,51);
           ellipseMode(CENTER);
           ellipse(pos.x,pos.y,5,5); 
        }
        if(bodyFound) {
           noStroke();
           fill(102,0,102);
           ellipseMode(CENTER);
           ellipse(pos.x,pos.y,5,5); 
        }
      }
      pos.add(direction);
      distance +=1;
    }
    if(replay && seeVision) {
       noStroke();
       fill(0,255,0);
       ellipseMode(CENTER);
       ellipse(pos.x,pos.y,5,5); 
    }
    look[2] = 1/distance;
    return look;
  }
  
  public void think() {  //think about what direction to move
      decision = brain.output(vision);
      int maxIndex = 0;
      float max = 0;
      for(int i = 0; i < decision.length; i++) {
        if(decision[i] > max) {
          max = decision[i];
          maxIndex = i;
        }
      }
      
      switch(maxIndex) {
         case 0:
           moveUp();
           break;
         case 1:
           moveDown();
           break;
         case 2:
           moveLeft();
           break;
         case 3: 
           moveRight();
           break;
      }
  }
  
  public void moveUp() { 
    if(yVel!=SIZE) {
      xVel = 0; yVel = -SIZE;
    }
  }
  public void moveDown() { 
    if(yVel!=-SIZE) {
      xVel = 0; yVel = SIZE; 
    }
  }
  public void moveLeft() { 
    if(xVel!=SIZE) {
      xVel = -SIZE; yVel = 0; 
    }
  }
  public void moveRight() { 
    if(xVel!=-SIZE) {
      xVel = SIZE; yVel = 0;
    }
  }
}
  static public void main(String[] passedArgs) {
    String[] appletArgs = new String[] { "--present", "--window-color=#666666", "--stop-color=#cccccc", "SnakeAI2" };
    if (passedArgs != null) {
      PApplet.main(concat(appletArgs, passedArgs));
    } else {
      PApplet.main(appletArgs);
    }
  }
}
