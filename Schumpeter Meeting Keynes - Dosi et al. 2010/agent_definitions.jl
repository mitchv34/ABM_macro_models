using Agents
using Distributions
using Distances



# Start by defining the agents populating the economy

mutable struct MPFirms <: AbstractAgent
    id::Int
    pos::NTuple{2,Int}
    type::String
    # Capital-good firm specific parameters
    A_t::Float32 # Output of the firm
    B_t::Float32 # Input of the firm
    p_t::Float32 # Price that the firm sets
    S::Float32 # Past sales of the firm
    RD::Float32 # R&D investme of the firm
    constumers::Vector{Int64} # List of consumers of the firm
    brochure::Vector{Int64} # List of CGFirm that the firm will offer their products
end

mutable struct CGFirms <: AbstractAgent
    id::Int
    pos::Dims{2}
    type::String
    # Consumption-good firm specific parameters
    ED::Float32 # Expected Demand
    D::Vector{Float32} # Demand history of the firm
    N_j::Float32 # Stock of inventory of the firm
    Q_t::Float32 # Output of the firm
    # ? TODO: I will interpret capital stock of the firm as the machines that the firm has
    K_t ::Float32 # Capital stock of the firm
    Ξ ::Vector{Float32} # Vintages machine tools belonging to the firm
    costumer_of ::Vector{Int64} # List of capital good firms that sell machine tools to the firm
end

mutable struct Consumers <: AbstractAgent
    id::Int
    pos::Dims{2}
end


# Now we define the economy

# We define the parameters of the economy
parameters = Dict(
                :F₁ => 50, # Number of firms in capital-good industry
                :F₂ => 200, # Number of firms in consumption-good industry
                :μ₁ => 0.04, # Capital-good firm mark-up rule
                :ν => 0.04, # R&D investment propensity
                :ξ => 0.50,# R&D allocation to innovative search
                :ς₁ => 0.30,# Capital-good firm search capabilities parameters
                :ς₂ => 0.30,# Consumption-good firm search capabilities parameters
                :dist_inn_proc =>  Beta(3,3)*(.15-(-.15))+(-.15), # Distribution (innovation process) α₁ = 3, β₁ = 3, ̲x₁ =-0.15, ̅x₁= 0.15

                :b => 3 , # Payback period
                :γ => 0.50, # New-customer sample parameter
                :ι => 0.10, # Desired inventories of consumption-good firms
)


# and the aggreate variables
agg_vars = Dict(:w => [], # Wage rate
)

# Define the economy as an ABM model that recieves a list of parameters 
space = GridSpace((1,1), periodic = false)
economy = AgentBasedModel(Union{ MPFirms, CGFirms}, space; scheduler = property_activation(:id))

# Next we define a function to populate the economy that recieves the number of firms in each industry and creates the agents  



function populate_economy!(economy, parameters)
    # We ad an agent to the economy for each firm in the capital-good industry
    for i ∈ 1:parameters[:F₁]
        # We add Machine Production Firms with random productivity parameter from a normal bivariate distribution
        A, B = rand(MvNormal([2, 2], [0.1 0.01; 0.01 0.1]))
        agent = agent = MPFirms(i, (1,1), "MPFirm", A, B, 0, 0, 0, [], [])
        add_agent!(agent, economy)
    end
    # for i ∈ 1+parameters[:F₁]:parameters[:F₂]+parameters[:F₁]
    #     agent = CGFirms(i, (1,1), "CGFirm", 0.0, 0.0, 0.0, 0.0, [], [])
    #     add_agent!(agent, economy)
    # end
end

# Now we define a step function to determine what happens to an agent when activated.

function step_economy!(agent, economy, parameters, agg_vars,)
    # First we determione what kind of agent it is
    if agent.type == "MPFirm"
            # If it is a firm in the capital-good industry, we calculate costs and prices using a simple markup rule
            c = w / agent.B_t;
            p = (1 + parameters[:μ₁]) * c
    
            # Next we calculate the firms investment in R&D as a fraction of past sales and the split between innovation and imitation
            agent.RD = parameters[:ν] * agent.S
            IN = parameters[:ξ] * agent.RD
            IM = (1 - parameters[:ξ]) * agent.RD
    
            # Now we determine if the firm will have acces to innovation with a realization of a Bernoulli trial
            θ_IN = 1 - exp(-parameters[:ς₁] * IN)
            if rand(Bernoulli(θ_IN))
                # If the firm has access to innovation, they draw a new machine characterized by a new A and B acording to:
                A_IN = agent.A_t*(1+rand(parameters[:dist_inn_proc]))
                B_IN = agent.B_t*(1+rand(parameters[:dist_inn_proc]))
            else   
                A_IN = 0
                B_IN = 0
            end
            
            # Now we determine if the firm will have access to the same machine as another firm in the same industry 
            θ_IM = 1 - exp(-parameters[:ς₂] * IM)
            if rand(Bernoulli(θ_IM))
                # If Firms have access to imitation they:
                # TODO: For now just imitating the most similar firm, need to implement a more sophisticated mechanism
                
                # Find the closest firm to the current firm
                tech = zeros(2, parameters[:F₁]) # Initialize a matrix to store the the technologies of the all firms
                j = 1
                for firm in economy.agents # Fill the matrxi with the technologies of all firms
                    if firm[2].type == "MPFirm"
                        tech[:, j] = [firm[2].A_t, firm[2].B_t]
                        j += 1
                    end
                end
                similarity =  sum( ([agent.A_t, agent.B_t] .- tech).^2 , dims=1) # Calculate the similarity between the current firm and all other firms
                similarity = sqrt.([s for s in similarity if s != 0]) # Remove the 0 similarity 
                closest_tech  = tech[:, sortperm( similarity )[1]] # Find the technology of the closest firm
    
                A_IM, B_IM = closest_tech # Use the technology of the closest firm
            else
                A_IM, B_IM = (0, 0)
            end                
    
            # Now we determine which machine the firm will produce from the three options: legacy, innovation or imitation
    
            machine_options = agg_vars[:w] .* [(1 + parameters[:μ₁])/agent.B_t + b/agent.A_t,
                             (1 + parameters[:μ₁])/B_IN + b/A_IN, 
                             (1 + parameters[:μ₁])/B_IM + b/A_IM]
    
            best_option = argmin(machine_options) # Find the best option
            if best_option == 2 # Based on what the best option is we update the firm technology
                agent.A_t = A_IN
                agent.B_t = B_IN
            elseif best_option == 3
                agent.A_t = A_IM
                agent.B_t = B_IM
            end
            
            # We update the price of the firm
            agent.p_t =  (1 + parameters[:μ₁])/agent.B_t

            # Now we determine a list of potencial new customers
            cg_ind = [cg_f[2].id for cg_f in economy.agents if cg_f[2].type == "CGFirm" && cg_f[2].id ∉ agent.constumers]
    
            # The firm offers their new machine to both their historical clients and a random sample of new clients
            agent.brochure = append!( agent.constumers, rand( cg_ind , Int( floor( parameters[:γ] * lenght(agent.constumers) ) ) ) )
    
    elseif agent.type == "CGFirm"
        # If it is a firm in the cconsumption-good industry, we estimate the expected demand using last period actual demand
        agent.ED = agent.D[end]
        # Calculate the desired inventories of the firm based on expected demand
        N_desired = agent.ED * parameters[:ι]
        # Calculate the desired level of production based on the expected demand and desired inventories and existing inventories
        agent.Q_t = agent.ED + N_desired - agent.N
        
        # Update capital stock of the firm        # We determine the new A and B of the firm
        agent.K_t = length(agent.Ξ)
        # Next we check if the firm will expand it's capital stock
        # If the firm does not have enough to meet it's expected demand, we determine how much it will add to it's capital stock
        EI = (agent.Q_t > agent.K_t) ? agent.Q_t - agent.K_t : 0

        # Now we determine which machine firms are sending brochures to the firm
        mg_id = [mg_f[2].id for mg_f in economy.agents if mg_f[2].type == "MPFirm" && agent.id ∈ mg_f[2].brochure]
        
        # Now we determine the cost and the unit price of each new machine offerered to the firm
        
        mach_cost = [agg_vars[:w] .* ( (1 + parameters[:μ₁])/economy.agents[i].B_t + b/economy.agents[i].A_t)  for i in mg_id]
        
        # ? TODO: This process of adding to capital stock and machinery replacing need a sanity check

        # First the firm will add to its capital stock the most cost effective machine
        
        machines_to_expand = []
        while EI > 0
            # Find the most cost effective machine
            indx = argmin(mach_cost)
            best_firm = mg_id[indx]
            # Add the most cost effective machine to the capital stock
            push!(machines_to_expand, (period, economy.agents))
            
        end
        # Next we determine which machines will the firm replace
        for i in mg_id
        end 
        agent.Ξ 

        # The firm replaces machine
    end
end





populate_economy!(economy, parameters)

economy.agents

economy.agents[1]

tech[:, sortperm( similarity )[1]]
