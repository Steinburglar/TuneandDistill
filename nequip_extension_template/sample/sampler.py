#provides a class that does sampling



class Sampler:
    def __init__(self, calculator, n_samples, output_path):
        self.calculator = calculator
        self.n_samples = n_samples
        self.output_path = output_path

    def generate(self):
        #actually generates the data. results in saved data at output_path
        raise NotImplementedError("Subclasses must implement the sample method")
    
        #need to ask claude about how to build a generator that saves state and "yeilds"
    
    
    @classmethod
    def from_sampler_checkpoint():
        #restores a sampler from a checkpointed sampler object
        #should load any persistent state of the sampler, as well as at least a path to the sampled frames
        #perhaps should not hold the data itself in memory, but an output datamodule that points to and organizes the data
        
        raise NotImplementedError("Subclasses must implement the from_sampler_checkpoint method")
