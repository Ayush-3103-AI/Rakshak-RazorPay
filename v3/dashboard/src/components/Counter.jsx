// A number that counts up once, on entry (#79).
//
// The value is always the artifact's value: the animation interpolates the way
// there and never the destination, so a reader who lands mid-tween and one who
// lands after it read the same figure. Under reduced motion, and when the value
// is not a finite number, it renders the final text immediately — a counter that
// animates toward "—" is noise, and a missing metric must stay visibly missing.
import { animate, useInView, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

export default function Counter({ value, format = (v) => String(v), duration = 1.1, className }) {
  const ref = useRef(null);
  const seen = useInView(ref, { once: true, margin: "-60px" });
  const reduce = useReducedMotion();
  const animatable = typeof value === "number" && Number.isFinite(value);
  const [shown, setShown] = useState(animatable && !reduce ? 0 : value);

  useEffect(() => {
    if (!animatable || reduce) {
      setShown(value);
      return undefined;
    }
    if (!seen) return undefined;
    const controls = animate(0, value, {
      duration,
      ease: [0, 0, 0.2, 1], // Blade's entrance easing
      onUpdate: setShown,
      onComplete: () => setShown(value), // land exactly on the artifact's value
    });
    return () => controls.stop();
  }, [seen, value, duration, reduce, animatable]);

  // `data-counter` so a test can find exactly these and nothing else. Querying by
  // the `tabular-nums` class instead matches every mono figure on the page, which
  // is how a reduced-motion assertion passes on a journey literal while every
  // counter it was meant to check is still sitting at zero.
  return (
    <span ref={ref} data-counter="" className={className}>
      {format(animatable ? shown : value)}
    </span>
  );
}
