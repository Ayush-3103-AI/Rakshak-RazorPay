// Thin wrapper over Radix's tooltip primitive — correct keyboard focus and
// ARIA for free, restyled to the Blade token set. Used on chart points,
// truncated hashes, and citation lists where the full value matters but
// would break the density budget if always shown.
import * as RadixTooltip from "@radix-ui/react-tooltip";

export function TooltipProvider({ children }) {
  return (
    <RadixTooltip.Provider delayDuration={200} skipDelayDuration={100}>
      {children}
    </RadixTooltip.Provider>
  );
}

export function Tooltip({ content, children }) {
  if (!content) return children;
  return (
    <RadixTooltip.Root>
      <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
      <RadixTooltip.Portal>
        <RadixTooltip.Content
          sideOffset={6}
          className="z-50 max-w-xs rounded-[var(--radius-sm)] border border-border bg-card px-[var(--spacing-4)] py-[var(--spacing-3)] text-xs text-card-foreground shadow-[var(--shadow-mid)]"
        >
          {content}
          <RadixTooltip.Arrow className="fill-card" />
        </RadixTooltip.Content>
      </RadixTooltip.Portal>
    </RadixTooltip.Root>
  );
}
