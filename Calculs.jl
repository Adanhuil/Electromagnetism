import Pkg
Pkg.add("ForwardDiff")
Pkg.add("GLMakie")
Pkg.add("LinearAlgebra")
Pkg.add("ColorSchemes")

using ForwardDiff
using GLMakie
using LinearAlgebra
using ColorSchemes

set_theme!(theme_black())

function ScalarField(V,xlims=[-3,3]::AbstractArray, ylims=[-3,3]::AbstractArray, steps=100::Int;sliderLimits=[]::Vector)
    set_theme!(theme_black())
    nSliders = length(sliderLimits)
    local sliders,sliderobservables,surfacePoints

    fig = Figure();
    ax1 = Axis3(fig[1, 1])
    xs = LinRange(xlims..., steps)
    ys = LinRange(ylims..., steps)

    if nSliders > 0
        sliders = [Slider(fig[1,i+1], range=sliderLimits[i], horizontal=false, height=350, label=string(i)) for i in 1:nSliders]
        [Label(fig[1,i+1],string(i),color=:black) for i in 1:nSliders]
        sliderobservables = [s.value for s in sliders]
        for i in 1:nSliders
            set_close_to!(sliders[i],(sliderLimits[i][end]-sliderLimits[i][1])/2)
        end
        surfacePoints = lift(sliderobservables...) do a...
            [V(x,y,a...) for x in xs, y in ys]
        end
    else
        ∇V(r) = ForwardDiff.gradient(V,r)
        ax2 = Axis(fig[1,2],xlabel="x",ylabel="y")
        minField = minimum(V(x,y) for x in xs for y in ys)
        surfacePoints = [V(x,y) for x in xs, y in ys]
        stream(x,y) = Point(∇V([x,y])...)
        streamplot!(ax1,stream,xlims[1]..xlims[2],ylims[1]..ylims[2],transformation = (:xy, minField))
        streamplot!(ax2,stream,xlims[1]..xlims[2],ylims[1]..ylims[2])
    end

    surface!(ax1,xs, ys, surfacePoints, colormap=:viridis)
    display(fig)
end

########################################################################
#                                C7
########################################################################

function C7()
    V(x,y) = x*exp(-(x^2+y^2))
    V(r) = V(r[1],r[2])
    ∇V(r) = ForwardDiff.gradient(V,r)

    n1(x,y) = 1/√3*[1.,1.]
    n2(x,y) = [-sin(atan(y,x)),cos(atan(y,x))]
    n1_∇V(x,y) = ∇V([x,y])'*n1(x,y)
    n2_∇V(x,y) = ∇V([x,y])'*n2(x,y)

    fig = Figure();
    cmap = :haline
    cmap2 = cgrad(colorschemes[:ice], 1.0:-0.01:0.0, categorical = true)[50:80];

    xs = LinRange(-2, 2, 1000)
    ys = LinRange(-2, 2, 1000)

    # Plots of the potential and its gradient (left of the figure)
    ax11 = Axis(fig[1,1],xlabel="x",ylabel="y",title=L"Gradient de $V$")
    stream1(x,y) = Point(∇V([x,y])...)
    streamplot!(ax11,stream1,-2..2,-2..2,colormap=cmap)

    ax21 = Axis3(fig[2,1],title=L"$V$")
    surface!(ax21,xs, ys, V, colormap=cmap);
    minField = minimum(V(x,y) for x in xs for y in ys)
    streamplot!(ax21,stream1,-2..2,-2..2,transformation = (:xy, minField),colormap=cmap)


    # Plots of the vector n and its product with the gradient of the potential (middle of the figure)
    ax12 = Axis(fig[1,2],xlabel="x",ylabel="y",title=L"Champ de vecteur $\mathbf{n}$")
    stream2(x,y) = Point(n1(x,y)...)
    streamplot!(ax12,stream2,-1..1,-1..1,arrow_size=15,density=0.7,colormap=cmap2)
    ax22 = Axis3(fig[2,2], viewmode=:fitzoom,title=L"Gradient de $V$ dans la direction de $\mathbf{n}$")
    surface!(ax22,xs,ys,n1_∇V,colormap=cmap)

    # Plots of the vector n' and its product with the gradient of the potential (middle of the figure)
    ax13 = Axis(fig[1,3],xlabel="x",ylabel="y",title=L"Champ de vecteur $\mathbf{n'}$")
    stream3(x,y) = Point(n2(x,y)...)
    streamplot!(ax13,stream3,-1..1,-1..1,arrow_size=15,density=0.7,colormap=cmap2)
    ax23 = Axis3(fig[2,3], viewmode=:fitzoom,title=L"Gradient de $V$ dans la direction de $\mathbf{n'}$")
    surface!(ax23,xs,ys,n2_∇V,colormap=cmap)

    display(fig)
end

C7()


########################################################################
#                                C8
########################################################################

V(x,y) = 10(2x*y - 3x^2 - 4y^2 - 18x + 28y + 12)
V(r) = V(r[1],r[2])
ScalarField(V,[-20,20],[-20,20],1000)

V(x,y,a,b,c) = 10(c*x*y - a*x^2 - b*y^2 - 18x + 28y + 12)
ScalarField(V,[-50,50],[-50,50],sliderLimits=[-6:0.01:6,-8:0.01:8,-4:0.01:4])


########################################################################
#                               C11
########################################################################

A(x,y,z) = [-x,y,0]
A(r::Vector) = A(r[1],r[2],r[3])
∇A(x,y,z) = ForwardDiff.jacobian(A,[x,y,z])

xs = LinRange(-10, 10, 100)
ys = LinRange(-10, 10, 100)
zs = LinRange(-10, 10, 100)

# Divergence

divergence = [∇A(x,y,z)[1,1] + ∇A(x,y,z)[2,2] + ∇A(x,y,z)[3,3] for x in xs, y in ys, z in zs]

fig = Figure();
ax = Axis3(fig[1,1])
surface!(ax,xs,ys,divergence[:,:,1])
display(fig)

# Rotational

rotational = [[∇A(x,y,z)[2,3]-∇A(x,y,z)[3,2],∇A(x,y,z)[3,1]-∇A(x,y,z)[1,3],∇A(x,y,z)[1,2]-∇A(x,y,z)[2,1]] for x in xs, y in ys, z in zs]
norms = [norm(rotational[i,j,k]) for i in 1:length(xs), j in 1:length(ys), k in 1:length(zs)]

fig = Figure();
ax = Axis3(fig[1,1])
surface!(ax,xs,ys,norms[:,:,1])
display(fig)

########################################################################
#                               C12
########################################################################

A(x,y) = Point2([-y/(x^2+y^2),x/(x^2+y^2)])

cmap = :viridis
fig = Figure(resolution=(1400,1400),fontsize = 40);
axis = Axis(fig[1,1],xlabel="x",ylabel="y",title=L"Champ vectoriel $\mathbf{A}$ et contour $C$")
streamplot!(axis,A,-0.5..0.5,-0.5..0.5,colormap=cmap,arrow_size=20,linewidth=3)

norms = [norm(A(x,y)) for x in 0.001:0.001:0.5 for y in 0.001:0.001:0.5]
Colorbar(fig[1,2], limits = (minimum(norms), maximum(norms)), nsteps =100, colormap = cmap, ticksize=15, width = 15, tickalign=1)
xs = [0.3*cos(ϕ) for ϕ in 0:pi/50:2*pi+pi/50]
ys = [0.3*sin(ϕ) for ϕ in 0:pi/50:2*pi+pi/50]
lines!(xs,ys, linewidth=3)
display(fig)

########################################################################
#                               C15
########################################################################

# Plot a 3D sphere with a given number of meridians on the existing figure
function makeSphere(nMeridians=4::Int)
    mesh!(Sphere(Point3f(0), 1f0), color = (:white,0.5),transparency = true)
    ϕ = collect(LinRange(0,pi,nMeridians+1))
    deleteat!(ϕ,length(ϕ))
    azimuths = [cos.(ϕ),sin.(ϕ)]
    θ = LinRange(0,2*pi,100)
    lines!(cos.(θ),sin.(θ),0 .*θ, color=:gray, linewidth = 0.7, linestyle = :dash)
    for i in 1:nMeridians
        lines!(azimuths[1][i]*sin.(θ),azimuths[2][i]*sin.(θ),cos.(θ), color=:gray, linewidth = 0.7, linestyle = :dash)
    end
end

R = 1.0;
xs = [R*sin(θ)*cos(ϕ) for θ in 0:pi/15:pi for ϕ in 0:pi/15:2*pi]
ys = [R*sin(θ)*sin(ϕ) for θ in 0:pi/15:pi for ϕ in 0:pi/15:2*pi]
zs = [R*cos(θ) for θ in 0:pi/15:pi for ϕ in 0:pi/15:2*pi]

function A(x,y,z)
    θ = atan(√(x^2+y^2),z)
    ϕ = atan(y,x)
    return sin(2θ)*[sin(θ)*cos(ϕ),sin(θ)*sin(ϕ),cos(θ)]
end

As = [A(xs[i],ys[i],zs[i]) for i in 1:length(xs)]
Axs = [As[i][1] for i in 1:length(xs)]
Ays = [As[i][2] for i in 1:length(xs)]
Azs = [As[i][3] for i in 1:length(xs)]
norms = norm.(As)

cmap2 = cgrad(colorschemes[:amp], 1.0:-0.01:0.0, categorical = true)[40:100];
fig = Figure(resolution = (1400, 1400));
axis = Axis3(fig[1, 1])
makeSphere(3)
arrows!(axis, xs, ys, zs, Axs, Ays, Azs , arrowsize = 0.05, lengthscale = 0.2, linewidth = 0.02, arrowcolor = norms, linecolor = norms, colormap = cmap2)
Colorbar(fig[1,2], limits = (minimum(norms), maximum(norms)), nsteps =100, colormap = cmap2, ticksize=15, width = 15, tickalign=1)
display(fig)

########################################################################
#                               C19
########################################################################

F(x,y) = Point2([y*(4x^2+y^2),x*(2x^2+3y^2)])

cmap = :viridis
fig = Figure(resolution=(2800,1400),fontsize = 40);
ax1 = Axis(fig[1,1],xlabel="x",ylabel="y",title=L"Contour $C$")
xlims!(ax1,-0.4,0.4)
ylims!(ax1,-0.4,0.4)
a = 0.3;
b = 0.15;
xs = [a*sin(ϕ) for ϕ in 0:pi/50:2*pi+pi/50]
ys = [b*cos(ϕ) for ϕ in 0:pi/50:2*pi+pi/50]
lines!(ax1,xs,ys, linewidth=3)
xs = [a*sin(ϕ) for ϕ in 0:pi/8:2*pi]
ys = [b*cos(ϕ) for ϕ in 0:pi/8:2*pi]
l(x,y) = [a*sin(atan(y,x)),b*cos(atan(y,x))]
ls = [l(xs[i],ys[i]) for i in 1:length(xs)]
norms = norm.(ls)
lxs = [ls[i][1] for i in 1:length(xs)]
lys = [ls[i][2] for i in 1:length(xs)]
arrows!(ax1,zeros(length(xs)), zeros(length(xs)), lxs, lys, arrowsize = 20, lengthscale = 1.0, arrowcolor = :lightblue, linecolor = :lightblue)


ax2 = Axis(fig[1,2],xlabel="x",ylabel="y",title=L"Champs vectoriels $\mathbf{F}$ et $\frac{d\mathbf{r}^+}{dt}$")
streamplot!(ax2,F,-0.5..0.5,-0.5..0.5,colormap=cmap,arrow_size=20,linewidth=3)
xs = [a*sin(ϕ) for ϕ in 0:pi/50:2*pi+pi/50]
ys = [b*cos(ϕ) for ϕ in 0:pi/50:2*pi+pi/50]
lines!(ax2,xs,ys, linewidth=3)
xs = [a*sin(ϕ) for ϕ in 0:pi/8:2*pi]
ys = [b*cos(ϕ) for ϕ in 0:pi/8:2*pi]
dl(x,y) = [a*sin(atan(y,x)),-b*cos(atan(y,x))]
dls = [dl(xs[i],ys[i]) for i in 1:length(xs)]
norms = norm.(dls)
dlxs = [dls[i][1] for i in 1:length(xs)]
dlys = [dls[i][2] for i in 1:length(xs)]
arrows!(ax2,xs, ys, dlxs, dlys, arrowsize = 20, lengthscale = 0.3, arrowcolor = :lightblue, linecolor = :lightblue)

norms = [norm(F(x,y)) for x in 0.001:0.001:0.5 for y in 0.001:0.001:0.5]
Colorbar(fig[1,3], limits =(minimum(norms), maximum(norms)), nsteps =100, colormap = cmap, ticksize=15, width = 15, tickalign=1)

display(fig)