# Frontend Architecture & Styling Plan

## Architecture Overview

### Proposed Stack Transition

#### Current Stack (Simple but Limited)
```
- Vanilla JavaScript
- CSS with variables
- CDN dependencies
- No build process
- Manual state management
```

#### Target Stack (Modern & Scalable)
```
- React 18+ with TypeScript
- State: Zustand/Redux Toolkit
- Styling: Tailwind CSS + Styled Components
- Build: Vite
- Testing: Vitest + React Testing Library
- Components: Radix UI + Custom
```

## Component Architecture

### 1. Atomic Design Structure
```
src/
├── components/
│   ├── atoms/           # Basic elements
│   │   ├── Button/
│   │   ├── Input/
│   │   ├── Badge/
│   │   └── Icon/
│   ├── molecules/       # Compound components
│   │   ├── Card/
│   │   ├── FormField/
│   │   ├── StatCard/
│   │   └── ChartControls/
│   ├── organisms/       # Complex components
│   │   ├── ExperimentList/
│   │   ├── CooperationChart/
│   │   ├── NetworkVisualization/
│   │   └── StrategyAnalysis/
│   ├── templates/       # Page layouts
│   │   ├── DashboardLayout/
│   │   ├── AnalysisLayout/
│   │   └── ComparisonLayout/
│   └── pages/          # Full pages
│       ├── Overview/
│       ├── Experiments/
│       ├── Analysis/
│       └── Settings/
```

### 2. Component Examples

#### Atom: Button Component
```tsx
// components/atoms/Button/Button.tsx
import { styled } from '@/styles';
import { ButtonHTMLAttributes, forwardRef } from 'react';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

const StyledButton = styled.button<ButtonProps>`
  /* Base styles */
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 500;
  transition: all 150ms ease;
  cursor: pointer;
  
  /* Size variants */
  ${({ size }) => {
    switch (size) {
      case 'sm': return 'padding: 0.5rem 1rem; font-size: 0.875rem;';
      case 'lg': return 'padding: 0.75rem 1.5rem; font-size: 1.125rem;';
      default: return 'padding: 0.625rem 1.25rem; font-size: 1rem;';
    }
  }}
  
  /* Color variants */
  ${({ variant, theme }) => {
    switch (variant) {
      case 'primary':
        return `
          background: ${theme.colors.primary};
          color: white;
          &:hover { background: ${theme.colors.primaryDark}; }
        `;
      case 'secondary':
        return `
          background: ${theme.colors.secondary};
          color: white;
          &:hover { background: ${theme.colors.secondaryDark}; }
        `;
      default:
        return '';
    }
  }}
  
  /* Loading state */
  ${({ isLoading }) => isLoading && `
    pointer-events: none;
    opacity: 0.7;
  `}
`;

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ children, leftIcon, rightIcon, isLoading, ...props }, ref) => {
    return (
      <StyledButton ref={ref} {...props}>
        {isLoading && <Spinner />}
        {!isLoading && leftIcon}
        <span>{children}</span>
        {rightIcon}
      </StyledButton>
    );
  }
);
```

#### Molecule: Stat Card
```tsx
// components/molecules/StatCard/StatCard.tsx
interface StatCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon?: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  change,
  icon,
  trend = 'neutral'
}) => {
  return (
    <Card className="stat-card">
      <CardHeader>
        <Icon>{icon}</Icon>
        <Title>{title}</Title>
      </CardHeader>
      <CardBody>
        <Value>{value}</Value>
        {change !== undefined && (
          <Change trend={trend}>
            <TrendIcon trend={trend} />
            {Math.abs(change)}%
          </Change>
        )}
      </CardBody>
    </Card>
  );
};
```

#### Organism: Experiment Dashboard
```tsx
// components/organisms/ExperimentDashboard/ExperimentDashboard.tsx
export const ExperimentDashboard: React.FC = () => {
  const { experiments, isLoading } = useExperiments();
  const [filter, setFilter] = useState<FilterOptions>({});
  
  if (isLoading) return <DashboardSkeleton />;
  
  return (
    <DashboardContainer>
      <DashboardHeader>
        <Title>Experiments</Title>
        <Actions>
          <Button onClick={handleExport}>Export All</Button>
          <Button variant="primary" onClick={handleNew}>
            New Experiment
          </Button>
        </Actions>
      </DashboardHeader>
      
      <FilterBar onFilterChange={setFilter} />
      
      <ExperimentGrid>
        {experiments
          .filter(applyFilters(filter))
          .map(experiment => (
            <ExperimentCard
              key={experiment.id}
              experiment={experiment}
              onDelete={handleDelete}
              onAnalyze={handleAnalyze}
            />
          ))}
      </ExperimentGrid>
    </DashboardContainer>
  );
};
```

## State Management Architecture

### 1. Store Structure (Zustand)
```typescript
// stores/experimentStore.ts
interface ExperimentStore {
  // State
  experiments: Experiment[];
  currentExperiment: Experiment | null;
  filters: FilterOptions;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  loadExperiments: () => Promise<void>;
  addExperiment: (experiment: Experiment) => void;
  updateExperiment: (id: string, updates: Partial<Experiment>) => void;
  deleteExperiment: (id: string) => void;
  setCurrentExperiment: (id: string) => void;
  setFilters: (filters: FilterOptions) => void;
  
  // Computed
  filteredExperiments: () => Experiment[];
  experimentStats: () => ExperimentStats;
}

export const useExperimentStore = create<ExperimentStore>((set, get) => ({
  experiments: [],
  currentExperiment: null,
  filters: {},
  isLoading: false,
  error: null,
  
  loadExperiments: async () => {
    set({ isLoading: true, error: null });
    try {
      const experiments = await api.getExperiments();
      set({ experiments, isLoading: false });
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },
  
  // ... other actions
  
  filteredExperiments: () => {
    const { experiments, filters } = get();
    return applyFilters(experiments, filters);
  },
}));
```

### 2. Context Providers
```tsx
// contexts/ThemeContext.tsx
interface ThemeContextValue {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  accentColor: string;
  setAccentColor: (color: string) => void;
}

export const ThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [accentColor, setAccentColor] = useState('#3b82f6');
  
  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };
  
  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, accentColor, setAccentColor }}>
      <StyledThemeProvider theme={createTheme(theme, accentColor)}>
        {children}
      </StyledThemeProvider>
    </ThemeContext.Provider>
  );
};
```

## Styling System

### 1. Design Tokens
```typescript
// styles/tokens.ts
export const tokens = {
  colors: {
    // Brand colors
    primary: {
      50: '#eff6ff',
      100: '#dbeafe',
      200: '#bfdbfe',
      300: '#93c5fd',
      400: '#60a5fa',
      500: '#3b82f6',
      600: '#2563eb',
      700: '#1d4ed8',
      800: '#1e40af',
      900: '#1e3a8a',
    },
    
    // Semantic colors
    success: { /* ... */ },
    warning: { /* ... */ },
    error: { /* ... */ },
    
    // Neutral colors
    gray: { /* ... */ },
  },
  
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    '2xl': '3rem',
    '3xl': '4rem',
  },
  
  typography: {
    fonts: {
      sans: 'Inter, system-ui, sans-serif',
      mono: 'JetBrains Mono, monospace',
    },
    
    sizes: {
      xs: '0.75rem',
      sm: '0.875rem',
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem',
      '3xl': '1.875rem',
      '4xl': '2.25rem',
    },
    
    weights: {
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
  },
  
  radii: {
    sm: '0.25rem',
    md: '0.375rem',
    lg: '0.5rem',
    xl: '0.75rem',
    '2xl': '1rem',
    full: '9999px',
  },
  
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    base: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  },
  
  transitions: {
    fast: '150ms ease',
    base: '250ms ease',
    slow: '350ms ease',
  },
};
```

### 2. Theme Configuration
```typescript
// styles/theme.ts
export const createTheme = (mode: 'light' | 'dark', accent: string) => {
  const isDark = mode === 'dark';
  
  return {
    ...tokens,
    
    colors: {
      ...tokens.colors,
      
      // Dynamic theme colors
      background: isDark ? '#0f172a' : '#ffffff',
      surface: isDark ? '#1e293b' : '#f8fafc',
      surfaceHover: isDark ? '#334155' : '#f1f5f9',
      
      text: {
        primary: isDark ? '#f8fafc' : '#0f172a',
        secondary: isDark ? '#cbd5e1' : '#64748b',
        tertiary: isDark ? '#94a3b8' : '#94a3b8',
      },
      
      border: isDark ? '#334155' : '#e2e8f0',
      
      // Accent color
      accent,
      accentLight: lighten(0.1, accent),
      accentDark: darken(0.1, accent),
    },
    
    // Responsive breakpoints
    breakpoints: {
      xs: '0px',
      sm: '640px',
      md: '768px',
      lg: '1024px',
      xl: '1280px',
      '2xl': '1536px',
    },
  };
};
```

### 3. Global Styles
```typescript
// styles/global.ts
import { createGlobalStyle } from 'styled-components';

export const GlobalStyle = createGlobalStyle`
  /* CSS Reset */
  *, *::before, *::after {
    box-sizing: border-box;
  }
  
  * {
    margin: 0;
  }
  
  html {
    font-size: 16px;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  
  body {
    font-family: ${({ theme }) => theme.typography.fonts.sans};
    background-color: ${({ theme }) => theme.colors.background};
    color: ${({ theme }) => theme.colors.text.primary};
    line-height: 1.6;
    transition: background-color ${({ theme }) => theme.transitions.base};
  }
  
  /* Focus styles */
  :focus {
    outline: 2px solid ${({ theme }) => theme.colors.accent};
    outline-offset: 2px;
  }
  
  /* Scrollbar styles */
  ::-webkit-scrollbar {
    width: 12px;
    height: 12px;
  }
  
  ::-webkit-scrollbar-track {
    background: ${({ theme }) => theme.colors.surface};
  }
  
  ::-webkit-scrollbar-thumb {
    background: ${({ theme }) => theme.colors.border};
    border-radius: ${({ theme }) => theme.radii.md};
    
    &:hover {
      background: ${({ theme }) => theme.colors.text.tertiary};
    }
  }
  
  /* Animations */
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  
  @keyframes slideIn {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
`;
```

## Responsive Design System

### 1. Breakpoint Utilities
```typescript
// styles/responsive.ts
export const media = {
  xs: (styles: string) => `@media (min-width: 0px) { ${styles} }`,
  sm: (styles: string) => `@media (min-width: 640px) { ${styles} }`,
  md: (styles: string) => `@media (min-width: 768px) { ${styles} }`,
  lg: (styles: string) => `@media (min-width: 1024px) { ${styles} }`,
  xl: (styles: string) => `@media (min-width: 1280px) { ${styles} }`,
  '2xl': (styles: string) => `@media (min-width: 1536px) { ${styles} }`,
};

// Usage example
const Container = styled.div`
  padding: 1rem;
  
  ${media.md(`
    padding: 2rem;
  `)}
  
  ${media.lg(`
    padding: 3rem;
    max-width: 1200px;
    margin: 0 auto;
  `)}
`;
```

### 2. Responsive Grid System
```typescript
// components/layout/Grid.tsx
interface GridProps {
  cols?: number | { xs?: number; sm?: number; md?: number; lg?: number };
  gap?: string;
  align?: 'start' | 'center' | 'end' | 'stretch';
}

const Grid = styled.div<GridProps>`
  display: grid;
  gap: ${({ gap = '1rem' }) => gap};
  align-items: ${({ align = 'stretch' }) => align};
  
  ${({ cols }) => {
    if (typeof cols === 'number') {
      return `grid-template-columns: repeat(${cols}, 1fr);`;
    }
    
    return `
      grid-template-columns: repeat(${cols?.xs || 1}, 1fr);
      
      ${cols?.sm && media.sm(`grid-template-columns: repeat(${cols.sm}, 1fr);`)}
      ${cols?.md && media.md(`grid-template-columns: repeat(${cols.md}, 1fr);`)}
      ${cols?.lg && media.lg(`grid-template-columns: repeat(${cols.lg}, 1fr);`)}
    `;
  }}
`;
```

## Animation System

### 1. Motion Presets
```typescript
// styles/motion.ts
export const motion = {
  // Spring animations
  spring: {
    gentle: 'cubic-bezier(0.4, 0.0, 0.2, 1)',
    bouncy: 'cubic-bezier(0.68, -0.6, 0.32, 1.6)',
    smooth: 'cubic-bezier(0.25, 0.1, 0.25, 1.0)',
  },
  
  // Duration presets
  duration: {
    instant: '100ms',
    fast: '200ms',
    normal: '300ms',
    slow: '500ms',
  },
};

// Framer Motion variants
export const variants = {
  fadeIn: {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
  },
  
  slideUp: {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1 },
  },
  
  scaleIn: {
    hidden: { scale: 0.9, opacity: 0 },
    visible: { scale: 1, opacity: 1 },
  },
  
  stagger: {
    visible: {
      transition: {
        staggerChildren: 0.1,
      },
    },
  },
};
```

### 2. Interactive Animations
```tsx
// components/animations/InteractiveCard.tsx
const InteractiveCard = styled(motion.div)`
  background: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing.lg};
  cursor: pointer;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: ${({ theme }) => theme.shadows.lg};
  }
`;

export const AnimatedCard: React.FC = ({ children }) => {
  return (
    <InteractiveCard
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      {children}
    </InteractiveCard>
  );
};
```

## Performance Optimizations

### 1. Code Splitting
```typescript
// routes/index.tsx
const Overview = lazy(() => import('@/pages/Overview'));
const Experiments = lazy(() => import('@/pages/Experiments'));
const Analysis = lazy(() => import('@/pages/Analysis'));

export const routes = [
  { path: '/', element: <Overview /> },
  { path: '/experiments', element: <Experiments /> },
  { path: '/analysis/:id', element: <Analysis /> },
];
```

### 2. Memoization Strategy
```tsx
// components/ChartWrapper.tsx
export const ChartWrapper = memo(({ data, config }) => {
  const processedData = useMemo(() => 
    processChartData(data, config),
    [data, config]
  );
  
  const chartOptions = useMemo(() => 
    generateChartOptions(config),
    [config]
  );
  
  return <Chart data={processedData} options={chartOptions} />;
}, (prevProps, nextProps) => {
  // Custom comparison
  return isEqual(prevProps.data, nextProps.data) && 
         isEqual(prevProps.config, nextProps.config);
});
```

This comprehensive frontend architecture plan provides a solid foundation for building a modern, scalable web application for the N-Person Prisoner's Dilemma project.