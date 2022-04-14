import jax
from jax.experimental import host_callback as hcb
## Taken from https://github.com/google/jax/blob/main/tests/host_callback_test.py

def call_jax_other_device(jax_outside_fun, arg, *, device):
  """Calls a JAX function on a specific device with simple support for reverse AD.
  Functions whose name starts with "jax_outside" are called on another device,
  by way of hcb.call.
  """

  def run_jax_outside_fun(arg):
    return jax.jit(jax_outside_fun)(jax.device_put(arg, device))

  @jax.custom_vjp
  def make_call(arg):
    return hcb.call(run_jax_outside_fun, arg,
                    result_shape=jax.eval_shape(jax_outside_fun, arg))

  # Define the fwd and bwd custom_vjp functions
  def make_call_vjp_fwd(arg):
    # Return the primal argument as the residual. Use `make_call` for the
    # primal computation to enable higher-order AD.
    return make_call(arg), arg  # Return the primal argument as the residual

  def make_call_vjp_bwd(res, ct_res):
    arg = res  # residual is the primal argument

    def jax_outside_vjp_fun(arg_and_ct):
      arg, ct = arg_and_ct
      _, f_vjp = jax.vjp(jax_outside_fun, arg)
      ct_in, = f_vjp(ct)
      return ct_in

    return (call_jax_other_device(jax_outside_vjp_fun, (arg, ct_res), device=device),)

  make_call.defvjp(make_call_vjp_fwd, make_call_vjp_bwd)
  return make_call(arg)
