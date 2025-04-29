const std = @import("std");
const Im2Proc = @import("Im2Proc.zig");
const Numerals = @import("Complex.zig");

const Complex = Numerals.Complex;
const Convolve2D = Im2Proc.Convolve2D;
const Rnd = std.Random.DefaultPrng;
const Allocator = std.heap.page_allocator;
const FookMe = Im2Proc.FookMe;
const assert = std.debug.assert;

pub const ETA :f64 = 0.001618;
pub const FEATURES :usize = 2;

pub const GRAD_CLIP = 0.01618;

pub const Activation = *const fn([]Complex, []Complex) FookMe!void;

pub const Blueprints = struct {
    kernelCounts: []usize,
    kernelSizes: [][2]usize,
    layerActivations: []Activation,
    imgSize: [2]usize,
    imgDepth: usize,
};

pub const ViewBall = struct {

    malloc: @TypeOf(Allocator),
    sections: []Layer,
    arena: []Complex,
    inputs: []Complex,
    imgSize: [3]usize,
    err: f64,
    scratchSize: usize,

    pub fn init(incMal: @TypeOf(Allocator), prints: *Blueprints) !ViewBall {
        const layerCount = prints.kernelCounts.len;
        assert(layerCount == prints.kernelSizes.len and layerCount > 0);
        assert(layerCount == prints.layerActivations.len);

        var rnd = Rnd.init(19);

        var paramsTotal :usize = 0;

        var inputSizeRunner :[3]usize = .{prints.imgSize[0], prints.imgSize[1], prints.imgDepth};
        var kernelSizeRunner: [2]usize = .{prints.kernelSizes[0][0], prints.kernelSizes[0][1]};

        var scratchSize = (prints.imgSize[0] - kernelSizeRunner[0] + 1)
        * (prints.imgSize[1] - kernelSizeRunner[1] + 1) * prints.kernelCounts[0];

        paramsTotal = kernelSizeRunner[0]*kernelSizeRunner[1]*prints.kernelCounts[0];

        scratchSize = std.math.ceilPowerOfTwo(usize, 1024*64*64) catch @panic("Turkey");

        std.debug.print("ScratchSize: {} MB\n", .{scratchSize*@sizeOf(Complex)*(layerCount+1)/1024/1024});

        const scratchInit = try incMal.alloc(Complex, scratchSize * (layerCount+1));
        errdefer incMal.free(scratchInit);
        @memset(scratchInit, (Complex{.real=0.0,.imag=0.0}));

        var layers: []Layer = try incMal.alloc(Layer, layerCount);
        errdefer(incMal.free(layers));
        layers[0] = try Layer.init(0, inputSizeRunner, kernelSizeRunner, prints.kernelCounts[0],
                                   prints.layerActivations[0], scratchSize, &rnd, incMal);

        var runnerDataCount = prints.imgSize[0]*prints.imgSize[1]*prints.imgDepth + (prints.imgSize[0] + 1 - prints.kernelSizes[0][0])
                    * (prints.imgSize[1] + 1 - prints.kernelSizes[0][0]) * prints.kernelCounts[0];

        runnerDataCount += layers[0].outputSize[0] * layers[0].outputSize[1] * layers[0].kernelCount;

        for(1..layerCount) |l| {
            kernelSizeRunner[0] = prints.kernelSizes[l][0];
            kernelSizeRunner[1] = prints.kernelSizes[l][1];
            inputSizeRunner[0] = layers[l-1].outputSize[0];
            inputSizeRunner[1] = layers[l-1].outputSize[1];
            inputSizeRunner[2] = layers[l-1].kernelCount;

            paramsTotal += kernelSizeRunner[0]*kernelSizeRunner[1]*prints.kernelCounts[l];

            layers[l] = try Layer.init(l, inputSizeRunner, kernelSizeRunner,
                            prints.kernelCounts[l], prints.layerActivations[l], scratchSize, &rnd, incMal);

            runnerDataCount += layers[l].outputSize[0] * layers[l].outputSize[1] * layers[l].kernelCount;
        }

        const inputs = try incMal.alloc(Complex, runnerDataCount);

        return ViewBall {
            .malloc = incMal, .inputs = inputs, .sections = layers,
            .arena = scratchInit, .imgSize = .{prints.imgDepth, prints.imgSize[1], prints.imgSize[0]},
            .err = 0.0,
            .scratchSize = scratchSize,
        };
    }

    pub fn deinit(self: *ViewBall) void {
        for(0..self.sections.len) |s| {
            self.sections[s].deinit();
        }
        self.malloc.free(self.sections);
        self.malloc.free(self.arena);
        self.malloc.free(self.inputs);
    }

    pub fn eat(self: *ViewBall, img: []Complex) []Complex {

        var runner = img;
        runner = self.sections[0].process(runner, self.arena[0..self.scratchSize]);

        for(1..self.sections.len) |l| {
            runner = self.sections[l].process(runner, self.arena[(self.scratchSize*l)..((self.scratchSize*(l+1)))]);
        }

        return runner[0..FEATURES];

    }

    pub fn train(self: *ViewBall, img: []Complex, targets: []Complex) []Complex {

        var inputPtr: usize = img.len;
        var runnerPtr: usize = img.len;
        var runner = img;
        @memcpy(self.inputs[0..img.len], img);
        runner = self.sections[0].process(runner, self.arena[0..self.scratchSize]);
        runnerPtr += runner.len;

        for(1..self.sections.len) |l| {
            @memcpy(self.inputs[inputPtr..(inputPtr+runner.len)], runner);
            inputPtr += runner.len;
            runner = self.sections[l].process(runner, self.arena[(self.scratchSize*l)..((self.scratchSize*(l+1)))]);
            runnerPtr += runner.len;
        }
        @memcpy(self.inputs[inputPtr..(inputPtr+runner.len)], runner);

        assert(targets.len == runner.len);

        self.err = 0;

        for(0..targets.len) |t| {
            runner[t].real = runner[t].real - targets[t].real;
            self.err += std.math.pow(f64, runner[t].real, 2);

        }
        runnerPtr -= runner.len;

        //Isn't scaling this equivalent to scaling the learning rate
        self.err /= @as(f64, @floatFromInt(targets.len));

        var scratch = self.arena[(self.scratchSize*(self.sections.len-1))..(self.scratchSize*self.sections.len)];
        var backSize = targets.len;
        for(0..self.sections.len-1) |l| {
            inputPtr -= self.sections[self.sections.len-1-l].inputSize[0] * self.sections[self.sections.len-1-l].inputSize[1] * self.sections[self.sections.len-1-l].inputDepth;

            runner = self.sections[self.sections.len-1-l].backProp(self.inputs[inputPtr..(runnerPtr)], runner, scratch);

            scratch = self.arena[(self.scratchSize*(self.sections.len-1-(l+1)))..((self.scratchSize*(self.sections.len-(l+1))))];

            backSize = runner.len;
            runnerPtr -= backSize;
        }

        inputPtr -= self.sections[0].inputSize[0] * self.sections[0].inputSize[1] * self.sections[0].inputDepth;

        runner = self.sections[0].backProp(self.inputs[inputPtr..runnerPtr], runner, scratch);

        for(0..self.sections.len) |l| {
             self.sections[self.sections.len-1-l].update();
         }

        return runner;

    }


    const Layer = struct {

        malloc: @TypeOf(Allocator),
        kernels: []Kernel,
        outputs: []Complex,
        influence: []Complex,
        inputSize: [2]usize,
        kernelSize: [2]usize,
        outputSize: [2]usize,
        kernelCount: usize,
        inputDepth: usize,
        scratchSizeLower: usize,
        id: usize,

        pub fn init(ID: usize, inputSizes: [3]usize, kernelSizes: [2]usize, kernelCount: usize,
                    act: Activation, memSize: usize, rnd: *Rnd, incMal: @TypeOf(Allocator)) !Layer {

            var kernelGroup: []Kernel = try incMal.alloc(Kernel, kernelCount);
            errdefer(incMal.free(kernelGroup));
            for(0.. kernelCount) |k| {
                kernelGroup[k] = try Kernel.init(inputSizes[2], inputSizes[1], inputSizes[0], kernelSizes[1], kernelSizes[0], kernelCount, 1.0/@sqrt(@as(f64, @floatFromInt(3*inputSizes[0]*kernelSizes[0]*kernelSizes[1]))), act, incMal, rnd);
            }

            const outputSizes: [2]usize = .{inputSizes[0]+1-kernelSizes[0], inputSizes[1]+1-kernelSizes[1]};

            const outputs = try incMal.alloc(Complex, outputSizes[0] * outputSizes[1] * kernelCount);
            errdefer(incMal.free(outputs));


            const scratchSizeLower = memSize / kernelCount;

            const influence = try incMal.alloc(Complex, inputSizes[2] * inputSizes[1] * inputSizes[0]);
            errdefer(incMal.free(influence));

            return Layer{
                .malloc = incMal,
                .kernels = kernelGroup,
                .outputs = outputs,
                .influence = influence,
                .inputSize = .{inputSizes[0], inputSizes[1]},
                .kernelSize = .{kernelSizes[0], kernelSizes[1]},
                .outputSize = outputSizes,
                .kernelCount = kernelCount,
                .inputDepth = inputSizes[2],
                .scratchSizeLower = scratchSizeLower,
                .id = ID
            };

        }

        pub fn deinit(self: *Layer) void {
            for(0..self.kernels.len) |k| {
                self.kernels[k].deinit(self.malloc);
            }
            self.malloc.free(self.kernels);
            self.malloc.free(self.influence);
        }

        pub fn process(self: *Layer, img: []Complex, scratch: []Complex) []Complex {

            const scratchPK = scratch.len / self.kernelCount;

            var kRes = self.kernels[0].process(img, scratch[0..scratchPK]);
            @memcpy(self.outputs[0..self.outputSize[0]*self.outputSize[1]], kRes);
            for(1..self.kernels.len) |k| {
                kRes = self.kernels[k].process(img, scratch[(scratchPK*k)..(scratchPK*(k+1))]);
                @memcpy(
                    self.outputs[self.outputSize[0]*self.outputSize[1]*k..self.outputSize[0]*self.outputSize[1]*(k+1)],
                    kRes
                );
            }

            return self.outputs;

        }

        pub fn backProp(self: *Layer, sinner:[]Complex, errs: []Complex, scratch: []Complex) []Complex {

            const kernelSize = errs.len / self.kernelCount;

            for(0..self.kernels.len) |k| {
                const influencer = self.kernels[k].backProp(sinner, errs[(kernelSize*k)..(kernelSize*(k+1))], scratch);
                for(0..self.influence.len) |i| {
                    self.influence[i].addInPlace(influencer[i]);
                }
            }

            return self.influence;

        }

        pub fn update(self: *Layer) void {
            for(0..self.kernelCount) |k| {
                self.kernels[k].update();
            }
        }

        pub const Kernel = struct {

            inputs: []Complex,
            grads: [2][]Complex,
            weights: []Complex,
            weightDeltas: []Complex,
            outputs: []Complex,
            inputLen: usize,
            inDepth: usize,
            outputLen: usize,
            inWidth: usize,
            inHeight: usize,
            kWidth: usize,
            kHeight: usize,
            layerSize: usize,
            outWidth: usize,
            outHeight: usize,
            bias: Complex,
            deltaBias: Complex,
            act: Activation,

            pub fn init(inDepth: usize, inHeight: usize, inWidth: usize, height: usize,  width: usize, layerSize: usize, scalar: f64,
                        actUp: Activation, malloc: @TypeOf(Allocator), genDaddy: *std.Random.Xoshiro256) !Kernel
            {
                var gen = genDaddy;
                const datum = try malloc.alloc(Complex, height*width*inDepth);
                const delts = try malloc.alloc(Complex, height*width*inDepth);

                for(0..datum.len) |d| {
                    datum[d] = Complex { .real = 1 * 2.0*(gen.random().float(f64)-0.5) * scalar, .imag =  1 * 2.0*(gen.random().float(f64)-0.5) * scalar };
                }

                const inputSaver = try malloc.alloc(Complex, inDepth*inHeight*inWidth);

                const myBy = Complex{ .real=0.0, .imag=0.0 };

                const biasDelta = Complex{ .real=0.0, .imag=0.0 };

                const outHeight = (inHeight+1-height);
                const outWidth = (inWidth+1-height);

                const myGrads :[2][]Complex = .{try malloc.alloc(Complex, (inHeight+1-height)*(inWidth+1-width)), try malloc.alloc(Complex, (inHeight+1-height)*(inWidth+1-width))};

                const outputs = try malloc.alloc(Complex, outWidth*outHeight*inDepth);

                return Kernel{
                    .inputs = inputSaver, .grads = myGrads, .weights = datum, .weightDeltas = delts, .outputs = outputs,
                    .inputLen = inWidth*inHeight*inDepth, .inDepth = inDepth, .outputLen = height*width,
                    .inWidth = inWidth, .inHeight = inHeight, .outHeight = outHeight, .outWidth = outWidth,
                    .kWidth = width, .kHeight = height, .layerSize = layerSize,
                    .bias = myBy, .deltaBias = biasDelta, .act = actUp
                };
            }

            pub fn deinit(self: *Kernel, malloc: @TypeOf(Allocator)) void {
                malloc.free(self.weights);
                malloc.free(self.inputs);
                malloc.free(self.grads[0]);
                malloc.free(self.grads[1]);
                malloc.free(self.outputs);
                malloc.free(self.weightDeltas);
            }

            pub fn process(self: *Kernel, pic: []Complex, scratch: []Complex) []Complex {

                const offset = (scratch.len-64*(self.inDepth*self.outWidth*self.outHeight));

                const result = Convolve2D(pic, self.weights, self.inWidth, self.kHeight, self.kWidth, self.outHeight, self.outWidth, self.inDepth, self.inDepth, scratch[0..(offset)]);

                 for(0..self.inDepth-1) |z| {
                     Im2Proc.addInPlace(result[0..(self.outHeight*self.outWidth)], result[(self.outHeight*self.outWidth + self.outHeight*self.outWidth*(z))..(self.outHeight*self.outWidth + self.outHeight*self.outWidth*(z+1))]);
                 }

                const answer = result[0..(self.outHeight*self.outWidth)];

                Im2Proc.addBias(answer, self.bias);

                self.act(answer, scratch[offset..]) catch @panic("ProcFailed.. Loser");

                const grads1 = scratch[offset..(offset+self.outHeight*self.outWidth)];

                const grads2 = scratch[(offset+self.outHeight*self.outWidth)..((offset+2*self.outHeight*self.outWidth))];

                @memcpy(self.grads[0], grads1);
                @memcpy(self.grads[1], grads2);

                return answer;

            }

            pub fn backProp(self: *Kernel, sinner: []Complex, errs: []Complex, scratch: []Complex) []Complex {
                var sum = Complex { .real = 0.0, .imag = 0.0 };

                for(0..(self.outWidth*self.outHeight)) |xy| {
                    const contrib = self.grads[0][xy].multiply(errs[xy]);
                    var b = self.grads[1][xy];
                    const upc = errs[xy].negateImagFresh();
                    const contrib2 = b.negateImagFresh().multiply(upc);

                    errs[xy] = contrib.add(contrib2);
                    sum.addInPlace(errs[xy]);
                }


                for(0..self.inDepth) |d| {
                    for(0..self.outHeight) |y| {
                        for(0..self.outWidth) |x| {
                            self.outputs[d*self.outWidth*self.outHeight + y*self.outWidth + x] = errs[y*self.outWidth + x];
                        }
                    }
                }

                const conjScratchSize = self.kWidth * self.kHeight * self.inDepth;
                const deconvScratchSize = (scratch.len - conjScratchSize - sinner.len) >> 1;

                self.deltaBias.copyFrom(sum);
                self.deltaBias.scaleInPlace(1.0/@as(f64, @floatFromInt(self.layerSize)));

                const complexKernel = Im2Proc.complexConjFresh(self.weights, scratch[0..conjScratchSize]);

                const influence = Im2Proc.Convolve2D(complexKernel, self.outputs, self.kWidth, self.outHeight, self.outWidth, self.inHeight, self.inWidth, self.inDepth, self.inDepth, scratch[(conjScratchSize)..(conjScratchSize + deconvScratchSize)]);

                const inConj = Im2Proc.complexConjFresh(sinner, scratch[(conjScratchSize + deconvScratchSize)..(conjScratchSize + deconvScratchSize + sinner.len)]);

                const delts = Im2Proc.Convolve2D(inConj, self.outputs, self.inWidth, self.outHeight, self.outWidth, self.kHeight, self.kWidth, self.inDepth, self.inDepth, scratch[(conjScratchSize + deconvScratchSize + sinner.len)..(conjScratchSize + 2*deconvScratchSize + sinner.len)]);

                @memcpy(self.weightDeltas, delts);

                return influence;
            }

            pub fn update(self: *Kernel) void {

                for(0..self.weights.len) |w| {
                    self.weightDeltas[w].scaleInPlaceClip(ETA, GRAD_CLIP);
                    self.weights[w].subtractInPlace(self.weightDeltas[w]);
                }

                self.deltaBias.scaleInPlaceClip(ETA, GRAD_CLIP);
                self.bias.subtractInPlace(self.deltaBias);
            }

        };

    };

};

// BE VERY AWARE, THIS MAY DIVIDE BY ZERO IF IT DOESN'T LIKE YOU
pub fn LEAKY_RELU(a: []Complex, scratch: []Complex) FookMe!void {
    @setFloatMode(.optimized);
    const fairSize = scratch.len >> 1;
    for(0..a.len) |o| {
        var gradVal :f64 = 0.01;
        const val = @max(gradVal * a[o].real, a[o].real);
        gradVal = val / a[o].real;
        scratch[o] = Complex{
            .real = gradVal,
            .imag = gradVal,
        };
        scratch[o+fairSize] = Complex{.real=0.0, .imag=0.0};
        a[o].real = val;
        a[o].imag = gradVal * a[o].imag;
    }

}

//JUST MAKE SURE IT LIKES YOU AND IT MIGHT NOT DIVIDE BY ZERO.
pub fn RELU(a: []Complex, scratch: []Complex) FookMe!void {
    @setFloatMode(.optimized);
    const fairSize = scratch.len >> 1;
    for(0..a.len) |o| {
        var gradVal :f64 = 0.0;
        var val = Complex {.real = 0.0, .imag = 0.0};
        if(a[o].real > 0) {
            gradVal = 1.0;
            val.real = a[o].real;
            val.imag = a[o].imag;
        }
        scratch[o] = Complex{
            .real = gradVal,
            .imag = gradVal,
        };
        scratch[o+fairSize] = Complex{.real=0.0, .imag=0.0};
        a[o] = val;
    }

}

pub fn TANH(a: []Complex, scratch: []Complex) FookMe!void {
    @setFloatMode(.optimized);

    const fairSize = scratch.len >> 1;

    for(0..a.len) |o| {
        const mag = a[o].magnitude() + 1e-25;

        const u = a[o].getScaled(1.0/mag);
        const act = std.math.tanh(mag);
        const slope = 1 - act * act;
        const alpha = act / mag;
        const beta = slope-alpha;
        const dFZ = (alpha + slope) * 0.5;
        var uTwo = u.multiply(u);
        const dFZConj = uTwo.getScaled(beta*0.5);

        a[o] = u.getScaled(act);

        scratch[o] = Complex {.real=dFZ, .imag = 0.0 };
        scratch[o+fairSize] = dFZConj;
    }

}
