const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.heap.page_allocator;
const Numerals = @import("Complex.zig");
const Complex = Numerals.Complex;

pub const FookMe = error {
    ScratchedMisguidedly,
    EmptyInput,
    Overflow
};

pub fn complexScaleWriteForward(a: []Complex, b: Complex, scratch: []Complex) void {
    assert(a.len == b.len);

    for(0..a.len) |i| {
        scratch[i] = a.getScaled(b);
    }

}

pub fn complexConjFresh(a: []Complex, scratch:[]Complex) []Complex {
    for(0..a.len) |i| {
        scratch[i] = a[i].negateImagFresh();
    }
    return scratch[0..a.len];
}

pub fn complexConj(a: []Complex) void {
    for(0..a.len) |i| {
        a[i].negateImag();
    }
}

//Alot (((scratch.len-N) - (N))>>1) Complexes
pub fn fft1D(x: []Complex, width: usize, paddedWidth: usize, scratch: []Complex) []Complex {
    const N = (width + (width & 1)) >> 1;
        
    if ((paddedWidth) == 1) {
        return x;
    }
        
    assert(paddedWidth & (paddedWidth - 1) == 0);
    const even = scratch[0..((paddedWidth) >> 1)];
    const odd = scratch[((paddedWidth) >> 1)..(paddedWidth)];
        
    //Dynamic padding and splitting of signal x
    for (0..(N)) |rk| {
        even[rk] = x[2 * rk];
    }
    @memset(even[N..], Complex{ .real = 0.0, .imag = 0.0 });
    for (0..(width >> 1)) |rk| {
        odd[rk] = x[2 * rk + 1];
    }
    @memset(odd[(width >> 1)..], Complex{ .real = 0.0, .imag = 0.0 });
        
    var offset = (paddedWidth);
        
    var result = scratch[offset..(offset + paddedWidth)];
       
    offset += paddedWidth;
       
    const halfMarker = (((scratch.len - offset)) >> 1); // (((scratch.len-offset)&1)^1);
        
    const evenFFT = fft1D((even), N, paddedWidth >> 1, scratch[offset..(offset + halfMarker)]);
    const oddFFT = fft1D((odd), width>>1, paddedWidth >> 1, scratch[(offset + halfMarker)..(offset + 2 * halfMarker)]);
        
    for (0..((paddedWidth >> 1))) |k| {
        const waveNumber = @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(paddedWidth));
        const multiplier = Complex{ .real = @cos(2 * std.math.pi * waveNumber), .imag = -@sin(2 * std.math.pi * waveNumber) };
            
        const t = oddFFT[k].multiply(multiplier);
        result[k] = evenFFT[k].add(t);
        result[k + (paddedWidth >> 1)] = evenFFT[k].subtract(t);
    }
        
    return result;
}

pub fn ifft1D(x: []Complex, ogWidth: usize, paddedWidth: usize, scratch: []Complex) []Complex {
    const newX = complexConjFresh(x, scratch[0..x.len]);

    var y = fft1D(newX, paddedWidth, paddedWidth, scratch[newX.len..]);

    const lf = @as(f64, @floatFromInt(ogWidth));
    for (0..paddedWidth) |i| {
        y[i] = ((y[i])).negateImagFresh().divide(lf);
    }

    return y;
}

pub fn fft2DT(matrix: []Complex, rows: usize, cols: usize, depth: usize, paddedRowCount: usize, paddedColCount: usize, scratch: []Complex) []Complex {
    assert(matrix.len == rows * cols * depth);
    assert(paddedRowCount >= rows and paddedColCount >= cols);

    //INVESTIGATE ROW MAJOR
    var transposeish = scratch[0 .. paddedRowCount * paddedColCount * depth];
    @memset(transposeish, Complex{ .real = 0.0, .imag = 0.0 });

    const scratchReqPerRow = 2 * paddedColCount * @ctz(paddedColCount);

    for (0..depth) |z| {
        for (0..rows) |r| {
            const rowFFT = fft1D(matrix[((r * cols) + z * rows * cols)..((r + 1) * cols + z * rows * cols)], cols, paddedColCount, scratch[(paddedRowCount * paddedColCount * depth)..(paddedRowCount * paddedColCount * depth + scratchReqPerRow)]);
            for (0..paddedColCount) |c| {
                //AUTO-TRANSPOSE TO ROW MINOR
                //                 transposeish[c * paddedRowCount + r + z * paddedRowCount * paddedColCount] = rowFFT[c];
                transposeish[r * paddedColCount + c + z * paddedColCount * paddedRowCount] = rowFFT[c];
            }
        }
    }

    var transposeIsh = scratch[(depth * paddedRowCount * paddedColCount + scratchReqPerRow)..((depth * paddedRowCount * paddedColCount + scratchReqPerRow + paddedRowCount * paddedColCount * depth))];
    var transpose = nillerPoseSeveral(transposeish, depth, paddedRowCount, scratch[((depth * paddedRowCount * paddedColCount + scratchReqPerRow + paddedRowCount * paddedColCount * depth))..(((depth * paddedRowCount * paddedColCount + scratchReqPerRow + paddedRowCount * paddedColCount * depth)) + paddedRowCount * paddedColCount * depth)]);

    const scratchReqPerCol = 2 * paddedRowCount * @ctz(paddedRowCount);

    const workspace = scratch[(((depth * paddedRowCount * paddedColCount + scratchReqPerRow + paddedRowCount * paddedColCount * depth)) + paddedRowCount * paddedColCount * depth)..((((depth * paddedRowCount * paddedColCount + scratchReqPerRow + paddedRowCount * paddedColCount * depth)) + paddedRowCount * paddedColCount * depth) + scratchReqPerCol)];
    //INVESTIGATE ROW MINOR
    for (0..depth) |z| {
        for (0..(paddedColCount)) |c| {
            const colFFT = fft1D(transpose[(c * paddedRowCount + z * paddedRowCount * paddedColCount)..(c * paddedRowCount + paddedRowCount + z * paddedRowCount * paddedColCount)], paddedRowCount, paddedRowCount, workspace);
            for (0..paddedRowCount) |r| {
                //Switch to ROW MAJOR MAPPING
                //                 transposeIsh[r * paddedColCount + c + z * paddedRowCount * paddedColCount] = colFFT[r];

                //YO DUDE WE CAN PREVENT A HARD TRANSPOSITION IN THE IFFT BY LEAVING THIS TRANSPOSED (LEAST SIGNIFICANT: ROW)
                //THIS WORKS BECAUSE OF THE SPECIFIC USE CASE (Correlation)
                transposeIsh[c * paddedRowCount + r + z * paddedRowCount * paddedColCount] = colFFT[r];
            }
        }
    }

    return transposeIsh;
}

//(((rows-j)*cols)-i) mirror pattern?

pub fn ifft2D(matrix: []Complex, depth: usize, rows: usize, cols: usize, paddedRowCount: usize, paddedColCount: usize, scratch: []Complex) []Complex {
    //This takes like half the method time?
    //     var transposeish = nillerPoseSeveral(matrix, depth, paddedRowCount, scratch[0..(paddedRowCount * paddedColCount * depth)]);
    //const fuckall = scratch[(paddedRowCount * paddedColCount * depth) .. 2 * (paddedRowCount * paddedColCount * depth)];
    const fuckall = scratch[0..(paddedRowCount * paddedColCount * depth)];

    const scratchReqPerCol = 2 * paddedRowCount * @ctz(paddedRowCount) + paddedRowCount;
    //Investigate row-minor (Alas, we need the columns of the row-major format (that we seeemingly arbitrarily made), hence the transposition... again)
    for (0..depth) |z| {
        for (0..(paddedColCount)) |c| {
            const colFFT = ifft1D(matrix[((c * paddedRowCount) + z * paddedRowCount * paddedColCount)..((c + 1) * paddedRowCount + z * paddedRowCount * paddedColCount)], rows, paddedRowCount, scratch[(paddedRowCount * paddedColCount * depth)..(paddedColCount * paddedRowCount * depth + scratchReqPerCol)]);
            for (0..paddedRowCount) |r| {
                //Populate row-major to effectively transpose the row-minor mapping and avoid the manual transpose
                fuckall[r * paddedColCount + c + z * paddedRowCount * paddedColCount] = colFFT[r];
            }
        }
    }

    // We successfully avoided this!
    //const shitMyFaceOff = nillerPoseSeveral(fuckall, depth, paddedColCount, scratch[(2*paddedRowCount*paddedColCount*depth + 3*paddedRowCount)..(//(2*paddedRowCount*paddedColCount*depth + 3*paddedRowCount) + paddedRowCount*paddedColCount*depth)]);

    var offset = ((paddedRowCount * paddedColCount * depth + scratchReqPerCol));

    var transposeIsh = scratch[(offset)..(offset + paddedRowCount * paddedColCount * depth)];

    offset += paddedRowCount * paddedColCount * depth;

    const scratchReqPerRow = 2 * paddedColCount * @ctz(paddedColCount) + paddedColCount;
    //Investigate ROW MAJOR (
    for (0..depth) |z| {
        for (0..paddedRowCount) |r| {
            const rowFFT = ifft1D(fuckall[(r * paddedColCount + z * paddedRowCount * paddedColCount)..(r * paddedColCount + paddedColCount + z * paddedRowCount * paddedColCount)], cols, paddedColCount, scratch[offset..(offset + scratchReqPerRow)]);
            for (0..paddedColCount) |c| {
                //Maintain ROW MAJOR
                transposeIsh[r * paddedColCount + c + z * paddedRowCount * paddedColCount] = rowFFT[c];
            }
        }
    }

    //signal ready: ROW MAJOR

    return transposeIsh;
}
pub fn nillerPoseWriteForward(matrix: []Complex, scratch: []Complex, rows: usize) []Complex {

    const cols = matrix.len / rows;

    for(0..rows) |r| {
        for(0..cols) |c| {
            scratch[rows*(c) + r] = matrix[r*cols + c];
        }
    }

    return scratch[rows*cols];

}

pub fn nillerPoseSeveral(matrix: []Complex, depth: usize, rows: usize, scratch: []Complex) []Complex {

    const cols = matrix.len / depth / rows;

    for(0..depth) |i| {
    for(0..rows) |r| {
        for(0..cols) |c| {
            scratch[c*rows + r + i*rows*cols] = matrix[r*cols + c];
        }
    }
    }

    return scratch[0..rows*cols*depth];

}

pub fn hadamard(a: []Complex, b: []Complex) Complex {
    var sum :Complex = Complex {.real = 0.0, .imag = 0.0 };
    for(0..a.len) |i| {
        a[i] = a[i].multiply(b[i]);
        sum.addInPlace(a[i]);
    }
    return sum;
}

pub fn subtractInPlace(a: []Complex, b: []Complex) void {
    for(0..a.len) |i| {
        a[i].subtractInPlace(b[i]);
    }
}

pub fn addBias(a: []Complex, b: Complex) void {
    for(0..a.len) |i| {
        a[i].addInPlace(b);
    }
}

pub fn addInPlace(a: []Complex, b: []Complex) void {
    for(0..a.len) |i| {
        a[i].addInPlace(b[i]);
    }
}

pub fn scaleInPlace(a: []Complex, b: Complex) void {
    for(0..a.len) |i| {
        a[i].scaleInPlace(b);
    }
}

pub fn scaleInPlaceClip(a: []Complex, b: Complex, c: f64) void {
    for(0..a.len) |i| {
        a[i] = @max(-c, @min(c, a[i]));
        a[i].scaleInPlace(b);
    }
}

pub fn copyInto(a: []Complex, b: []Complex) void {
    for(0..a.len) |i| {
        a[i] = b[i];
    }
}

pub fn innerSum(a: []Complex) f64 {
    var sum: f64 = 0.0;
    for(a) |c| {
        sum += c.real;
    }
    return sum;
}

pub fn printRaw(matrix: []Complex, resHeight: usize) void {
    const resWidth = matrix.len / resHeight;
    if(resWidth < 1) return;
    for(0..resHeight) |j| {
        std.debug.print("\n\n[", .{});
        for(0..(resWidth-1)) |i| {
            std.debug.print("{}, ", .{matrix[j*resWidth + i].real});
        }
        std.debug.print("{}]\n", .{matrix[(resHeight-1)*resWidth + resWidth-1].real});
    }
}

pub fn print(matrix: []Complex, resHeight: usize) void {
    const resWidth = matrix.len/resHeight;
    for(0..resHeight) |j| {
        std.debug.print("\n\n[", .{});
        for(0..(resWidth-1)) |i| {
            std.debug.print("[{},{}]", .{matrix[j*resWidth + i].real, matrix[j*resWidth + i].imag});
        }
        std.debug.print("[{},{}]]\n", .{matrix[j*resWidth + resWidth-1].real, matrix[j*resWidth+resWidth-1].imag});
    }
}

pub fn printSeveral(matrix: []Complex, resDepth: usize, resHeight: usize) void {
    const resWidth = matrix.len/resDepth/resHeight;
    for(0..resDepth) |k| {
    for(0..resHeight) |j| {
        std.debug.print("\n\n[", .{});
        for(0..(resWidth-1)) |i| {
            std.debug.print("[{},{}]", .{matrix[j*resWidth + i + k*resHeight*resWidth].real, matrix[j*resWidth + i + k*resHeight*resWidth].imag});
        }
        std.debug.print("[{},{}]]\n", .{matrix[j*resWidth + resWidth-1 + k*resHeight*resWidth].real, matrix[j*resWidth+resWidth-1+k*resHeight*resWidth].imag});
    }
    }
}

pub fn indexOfMax(slice: []Complex) usize {
    assert(slice.len > 0);
    var best = slice[0];
    var index: usize = 0;
    for (slice[1..], 0..) |item, i| {
        if (item.real > best.real) {
            best = item;
            index = i + 1;
        }
    }
    return index;
}

pub fn Convolve2D(img :[]Complex, kernel: []Complex, imgWidth: usize, kernelHeight: usize, kernelWidth:usize, resHeight: usize, resWidth: usize, resDepth: usize, inDepth: usize, scratch: []Complex) []Complex {

    _ = resDepth;

    const imgHeight = img.len/inDepth/imgWidth;

    const maxH = @max(kernelHeight, @max(resHeight, imgHeight));
    const maxW = @max(kernelWidth, @max(resWidth, imgWidth));


    const scaledHeight = (std.math.ceilPowerOfTwo(usize, maxH) catch @panic("ShitMAN"));

    const scaledWidth = (std.math.ceilPowerOfTwo(usize, maxW) catch @panic("FUCK MCGEE"));

    var runningOffset = inDepth*(2*imgHeight*imgWidth + scaledHeight*scaledWidth) * 4;

    const ImageFFT = fft2D(img, imgHeight, imgWidth, inDepth, scaledHeight, scaledWidth, scratch[0..runningOffset]);

    const minAllocHeight = scaledHeight;
    const minAllocWidth = scaledWidth;

    const KernelFFT = fft2D(kernel, kernelHeight, kernelWidth, inDepth, scaledHeight, scaledWidth, scratch[runningOffset..(runningOffset + inDepth * (2*minAllocHeight*minAllocWidth + scaledHeight*scaledWidth) * 4)]);

    runningOffset = runningOffset + inDepth * (2*minAllocHeight*minAllocWidth + scaledHeight*scaledWidth) * 4;

    _ = hadamard(ImageFFT, KernelFFT);

    const convRes = ifft2D(ImageFFT, inDepth, scaledHeight, scaledWidth, scratch[runningOffset..(runningOffset+inDepth*(2*scaledHeight*scaledWidth + scaledHeight*scaledWidth) * 4)]);//(end+2*rows*cols+rows)]);

    runningOffset += inDepth*(2*scaledHeight*scaledWidth + scaledHeight*scaledWidth) * 4;

    const answer = scratch[runningOffset..(runningOffset+resHeight*resWidth*inDepth)];

    const extractY = (@intFromBool(imgHeight>=resHeight)) * (kernelHeight-1);
    const extractX = @intFromBool(imgWidth>=resWidth) * (kernelWidth-1);

    for(0..inDepth) |z| {
        for(0..resHeight) |y| {
            for(0..resWidth) |x| {
                answer[x + y*resWidth + z*resHeight*resWidth] = convRes[x + extractX + (y+extractY)*scaledWidth + z*scaledHeight*scaledWidth];
            }
        }
    }

    return  answer;

}
