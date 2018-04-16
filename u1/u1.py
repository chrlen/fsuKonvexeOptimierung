from myPlot import plotSurface
from myPlot import plotContour

from functions import rosenbrock
plotSurface(-10,10,0-1,-10,10,0.1,rosenbrock,"","rosenbrock")
plotContour(-10,10,0-1,-10,10,0.1,rosenbrock,"","rosenbrock")

from functions import himmelblau
plotSurface(-10,10,0-1,-10,10,0.1,himmelblau,"","himmelblau")
plotContour(-10,10,0-1,-10,10,0.1,himmelblau,"","himmelblau")

from functions import bazaraa_shetty
plotSurface(-10,10,0-1,-10,10,0.1,bazaraa_shetty,"","bazaraa_shetty")
plotContour(-10,10,0-1,-10,10,0.1,bazaraa_shetty,"","bazaraa_shetty")

from functions import beale
plotSurface(-10,10,0-1,-10,10,0.1,beale,"","beale")
plotContour(-10,10,0-1,-10,10,0.1,beale,"","beale")

from functions import spellucci
plotSurface(-10,10,0-1,-10,10,0.1,spellucci,"","spellucci")
plotContour(-10,10,0-1,-10,10,0.1,spellucci,"","spellucci")
