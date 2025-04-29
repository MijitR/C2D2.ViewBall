pub const Complex = struct {
    real: f64,
    imag: f64,

    pub inline fn add(self: Complex, adder: Complex) Complex {
        return Complex {
            .real = self.real + adder.real,
            .imag = self.imag + adder.imag
        };
    }

    pub inline fn addInPlace(self: *Complex, adder: Complex) void {
        self.real = self.real + adder.real;
        self.imag = self.imag + adder.imag;
    }

    pub inline fn addInPlaceAddr(self: *Complex, adder: *Complex) void {
        self.real = self.real + adder.*.real;
        self.imag = self.imag + adder.*.imag;
    }

    pub inline fn subtract(self: Complex, subber: Complex) Complex {
        return Complex {
            .real = self.real - subber.real,
            .imag = self.imag - subber.imag
        };
    }

    pub inline fn subtractInPlace(self: *Complex, adder: Complex) void {
        self.real = self.real - adder.real;
        self.imag = self.imag - adder.imag;
    }

    pub inline fn multiply(self: Complex, multer: Complex) Complex {
        return Complex {
            .real = self.real * multer.real - self.imag * multer.imag,
            .imag = self.real * multer.imag + self.imag * multer.real
        };
    }

    pub inline fn divide(self: Complex, value: f64) Complex {
        return Complex {
            .real = self.real / value,
            .imag = self.imag / value
        };
    }

    pub inline fn magnitude(self: Complex) f64 {
        return @sqrt(self.real*self.real + self.imag*self.imag);
    }

    pub inline fn getScaled(self: Complex, scalar: f64) Complex {
        return Complex {
            .real = self.real * scalar,
            .imag = self.imag * scalar
        };
    }

    pub inline fn scaleInPlace(self: *Complex, scalar: f64) void {
        self.real *= scalar;
        self.imag *= scalar;
    }

    pub fn scaleInPlaceClip(self: *Complex, b: f64, c: f64) void {
        self.real = @max(-c, @min(c, self.real));
        self.imag = @max(-c, @min(c, self.imag));
        self.scaleInPlace(b);
    }

    pub inline fn negate(self: *Complex) *Complex {
        self.real = -self.real;
        self.imag = -self.imag;
        return self;
    }

    pub inline fn negateImag(self: *Complex) void {
        self.imag = -self.imag;
    }

    pub inline fn negateFresh(self: *Complex) Complex {
        return Complex{
            .real = -self.real,
            .imag = -self.imag,
        };
    }

    pub inline fn negateImagFresh(self: *Complex) Complex {
        return Complex{
            .real = self.real,
            .imag = -self.imag,
        };
    }

    pub inline fn copyFrom(self: *Complex, b: Complex) void {
        self.real = b.real;
        self.imag = b.imag;
    }

};
