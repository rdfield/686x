use Modern::Perl;
use Data::Dumper;

my $T = [ 
          [ 
            [1/2,1/2,0,0,0], 
            [1/2,1/2,0,0,0], 
            [2/3,1/3,0,0,0]
          ], 
          [
             [1/3,2/3,0,0,0], 
             [1/4,1/2,1/4,0,0], 
             [0,2/3,1/3,0,0]
          ], 
          [
             [0,1/3,2/3,0,0], 
             [0,1/4,1/2,1/4,0], 
             [0,0,2/3,1/3,0]
          ], 
          [
             [0,0,1/3,2/3,0], 
             [0,0,1/4,1/2,1/4], 
             [0,0,0,2/3,1/3]
          ], 
          [
             [0,0,0,1/3,2/3], 
             [0,0,0,1/2,1/2], 
             [0,0,0,1/2,1/2]
          ], 
      ];

my $num_state = 5;
my $num_action = 3;
my $r = 1/2; 

my @V = (0) x $num_state; 
my @R = (0) x $num_state;
$R[-1] = 1;

my $num_iter = 100;

foreach my $i (0 .. $num_iter - 1) {
   my @Q;
   foreach my $t (0 .. $num_state - 1) {
      foreach my $a (0 .. $num_action - 1) {
         foreach my $s (0 .. $num_state - 1) {
#            say("Considering values for i = $i, t = $t, a = $a, s = $s");
#            say("Q += " . $T->[$s][$a][$t] . " * ( " . $R[$s] . " + " . $r . " * " . $V[$t] . ")");
#            say("Q += " . $T->[$s][$a][$t] * ($R[$s] + $r * $V[$t]) );
            $Q[$s][$a] += $T->[$s][$a][$t] * ($R[$s] + $r * $V[$t])
         }
      }
   }
   foreach my $row (0 .. $#Q) {
      my $max = 0;
      foreach my $col(0 .. $#{$Q[$row]}) {
         $max = $Q[$row][$col] if $Q[$row][$col] > $max;
      }
      $V[$row] = $max;
   }
}
say(Dumper(\@V));
__END__
# the original python
num_state = 5
num_action = 3
r = 1/2 
# initialization 
V = np.zeros(5) 
# reward 
R = np.zeros(5) 
R[4] = 1 
num_iter = 10 
for i in range(num_iter): 
   Q = [[sum([T[s][a][t] * (R[s] + r * V[t]) for t in range(num_state)]) for a in range(num_action)] for s in range(num_state)] 
   print("Q",Q)
   print("V before",V)
   V = np.max(Q, axis=1) 
   print("V after",V)
print(V) 
