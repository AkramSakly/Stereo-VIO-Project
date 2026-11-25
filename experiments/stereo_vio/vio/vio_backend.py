import numpy as np
from scipy.optimize import least_squares

class VIOBackend:
    def __init__(self, window_size=10):
        self.window_size = window_size
        # Initial state: Position, Velocity, Accel Bias
        self.states = [{
            'p': np.zeros((3, 1)),
            'v': np.zeros((3, 1)),
            'b_a': np.zeros((3, 1))
        }]
        # Gravity is a world-frame constant (Downwards on Z)
        self.gravity = np.array([[0], [0], [9.81]])
        
    def imu_preintegrate(self, imu_buffer, dt):
        """Summarizes high-frequency IMU data between camera frames."""
        delta_p = np.zeros((3, 1))
        delta_v = np.zeros((3, 1))
        b_a = self.states[-1]['b_a']
        
        for acc in imu_buffer:
            # Correct raw acceleration with current bias
            # Note: Gravity is handled in the global cost function
            a_corr = acc.reshape(3, 1) - b_a
            delta_p += delta_v * dt + 0.5 * a_corr * (dt**2)
            delta_v += a_corr * dt
            
        return delta_p, delta_v

    def cost_function(self, x, visual_anchor, preint_p, dt_total):
        """The core of the Tightly-Coupled VIO."""
        p_opt = x[:3].reshape(3, 1)
        v_opt = x[3:6].reshape(3, 1)
        
        p_prev = self.states[-1]['p']
        v_prev = self.states[-1]['v']
        
        # Predicted position based on physics: p_prev + v*dt + 1/2*(a - g)*dt^2
        # Using -self.gravity because we want to counteract the Earth's pull
        p_pred = p_prev + v_prev * dt_total + 0.5 * (-self.gravity) * (dt_total**2) + preint_p
        
        # Residual 1: Deviation from IMU physics
        res_imu = (p_opt - p_pred).flatten()
        
        # Residual 2: Deviation from Camera's PnP 'Anchor'
        res_vis = np.zeros(3)
        if visual_anchor is not None:
            res_vis = (p_opt - visual_anchor).flatten()
        
        # Combine residuals. Vision is weighted (2.5x) to maintain global scale.
        return np.concatenate([res_imu, res_vis * 2.5])

    def launch_optimization(self, visual_pose, imu_data, dt):
        """Executes the Huber-loss robust optimization."""
        if not self.states: return np.zeros((3, 1))
        
        dp, dv = self.imu_preintegrate(imu_data, dt)
        
        # Initial guess: Current state + IMU delta
        x0 = np.concatenate([
            (self.states[-1]['p'] + dp).flatten(),
            (self.states[-1]['v'] + dv).flatten()
        ])
        
        # Solve using Levenberg-Marquardt with Huber loss for outlier rejection
        res = least_squares(
            self.cost_function, 
            x0, 
            args=(visual_pose, dp, dt),
            loss='huber'
        )
        
        opt_p = res.x[:3].reshape(3, 1)
        opt_v = res.x[3:6].reshape(3, 1)
        
        # Update state history
        self.states.append({
            'p': opt_p, 'v': opt_v, 'b_a': self.states[-1]['b_a']
        })
        
        if len(self.states) > self.window_size:
            self.states.pop(0)
            
        return opt_p

if __name__ == "__main__":
    print("🔧 Testing VIO Backend Logic...")
    backend = VIOBackend()
    dummy_imu = [np.array([0.1, 0.0, 9.81]) for _ in range(10)]
    dummy_visual = np.array([[0.5], [0.1], [0.0]])
    
    try:
        result = backend.launch_optimization(dummy_visual, dummy_imu, dt=0.05)
        print(f"✅ Optimization Successful!")
        print(f"📍 New Position: {result.flatten()}")
    except Exception as e:
        print(f"❌ Error: {e}")