final int SIZE = 20;
final int hidden_nodes = 16;
final int hidden_layers = 2;
final int fps = 1500;  //15 is ideal for self play, increasing for AI does not directly increase speed, speed is dependant on processing power

int highscore = 0;
int score=0;
float mutationRate = 0.05;
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

void setup() {
  evolution = new ArrayList<Integer>();
  frameRate(fps);
  pop = new Population(2000);
}

void draw() {
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
  
  void grid(){
    fill(127);
      for(int i=0 ; i<50;i++){
        line(i*20,0,i*20,height);
        line(0,i*20,width,i*20);
      }
  }
  
  void keyPressed(){
    if (keyCode == UP )
          mutationRate += 0.05;
    if (keyCode == DOWN && mutationRate > 0.25 )
          mutationRate -= 0.05;
  
  }
